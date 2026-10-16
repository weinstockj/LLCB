
# =============================================================================
# Stage 1 of LLCB: interventional total effects with sampling uncertainty.
#
# The quantity stage 2 needs is the *hard-intervention* total effect
#
#     Θ̃[u, j] = d E[x_j | do(x_u = c)] / d c ,      normalised so Θ̃[u, u] = 1.
#
# Because every knockout sample has the perturbed gene set to a known value,
# this is a Wald (instrumental-variable) ratio: the donor-adjusted knockout
# contrast on readout j, divided by the donor-adjusted knockout contrast on the
# perturbed gene itself. Estimating it as a ratio of contrasts rather than as a
# pooled slope of x_j on x_u matters, because the pooled slope mixes the
# interventional contrast with the *within-control* covariance of u and j --
# precisely the confounded association the intervention was meant to bypass.
#
# All readouts for one knockout share a design matrix, so the whole row of Θ̃ is
# obtained from a single factorisation: O(p) work per knockout rather than the
# O(p^2) univariate regressions of the original implementation.
# =============================================================================

"""
    TotalEffectEstimate

Stage-1 estimate of the interventional total-effect matrix, together with the
sampling uncertainty that stage 2 needs in order to weight it.

# Fields
- `Θ::Matrix{Float64}`: `q × p`. `Θ[u, j]` is the total causal effect of a
  one-unit intervention on the `u`-th *perturbed* gene on readout `j`. The
  self-effect entry is stored as `0`; the implied `Θ̃` carries `1` there.
- `se::Matrix{Float64}`: delta-method standard errors of `Θ`.
- `Σ::Union{Nothing,Vector{Matrix{Float64}}}`: if `covariance = :full`, `Σ[u]`
  is the `p × p` sampling covariance of row `u` of `Θ`. Estimates within a row
  share control samples and are correlated; estimates in different rows share
  only the controls.
- `knockdown::Vector{Float64}`: the donor-adjusted contrast achieved on the
  perturbed gene itself (the Wald denominator).
- `knockdown_se::Vector{Float64}`: its standard error. `|knockdown / knockdown_se|`
  is the first-stage t statistic; see [`weak_knockdowns`](@ref).
- `active::BitMatrix`: entries that stage 2 should use (`false` on self-effects
  and on rows whose knockdown was too weak to identify a ratio).
- `nobs`, `dof`: samples and residual degrees of freedom per knockout.
- `perturbed::Vector{Int}`: column index of each perturbed gene.
- `targets::Vector{String}`: readout names, length `p`.
"""
struct TotalEffectEstimate
    Θ::Matrix{Float64}
    se::Matrix{Float64}
    Σ::Union{Nothing,Vector{Matrix{Float64}}}
    knockdown::Vector{Float64}
    knockdown_se::Vector{Float64}
    active::BitMatrix
    nobs::Vector{Int}
    dof::Vector{Int}
    perturbed::Vector{Int}
    targets::Vector{String}
end

# v0.3.0-dev positional form without `knockdown_se`, kept for callers that
# build estimates by hand (tests, simulations).
TotalEffectEstimate(Θ, se, Σ, knockdown, active, nobs, dof, perturbed, targets) =
    TotalEffectEstimate(Θ, se, Σ, knockdown, fill(NaN, length(knockdown)), active,
                        nobs, dof, perturbed, targets)

Base.size(te::TotalEffectEstimate) = size(te.Θ)

"""
    weak_knockdowns(te; threshold = sqrt(10)) -> Vector{Int}

Rows of `te` whose knockdown is weak in the instrumental-variable sense: the
first-stage t statistic `|knockdown / knockdown_se|` is below `threshold`
(the default is the usual first-stage F > 10 rule). Wald ratios in those rows
have heavy-tailed sampling distributions. Their standard errors are large, so
they carry little weight in [`fit_llcb`](@ref), but they can dominate any
unweighted use of the total effects, including
[`closed_form_coefficients`](@ref).
"""
function weak_knockdowns(te::TotalEffectEstimate; threshold::Real = sqrt(10))
    return [k for k in eachindex(te.knockdown)
            if isfinite(te.knockdown_se[k]) && abs(te.knockdown[k]) < threshold * te.knockdown_se[k]]
end

# Minimum-norm least squares via SVD. The design is tiny (intercept + knockout
# indicator + donor dummies), and donors can be collinear with the knockout
# indicator when a donor contributes no control samples, so rank deficiency has
# to be handled rather than assumed away.
function _svd_lsq(D::Matrix{Float64}, Y::Matrix{Float64})
    F = svd(D)
    tol = maximum(F.S) * max(size(D)...) * eps(Float64)
    sinv = [s > tol ? inv(s) : 0.0 for s in F.S]
    rank = count(>(0.0), sinv)
    C = F.V * (Diagonal(sinv) * (F.U' * Y))
    resid = Y - D * C
    return C, resid, rank, F, sinv
end

# (DᵀD)⁺[k, k] for the k-th regressor.
_leverage(F::SVD, sinv::Vector{Float64}, k::Int) =
    sum(abs2(F.V[k, i] * sinv[i]) for i in eachindex(sinv))

"""
    schafer_strimmer_shrinkage(R, dof)

Analytic shrinkage intensity for a covariance matrix estimated from residual
matrix `R`, shrinking the correlations toward zero (Schäfer & Strimmer 2005).
Replicates per knockout are far fewer than readout genes, so the unshrunk
residual covariance is singular and cannot be used for weighting as is.
"""
function schafer_strimmer_shrinkage(R::AbstractMatrix{Float64}, dof::Int)
    n, p = size(R)
    (n < 3 || p < 2) && return 1.0
    s = vec(sqrt.(sum(abs2, R; dims = 1) ./ dof))
    s[s .<= 0] .= Inf                       # constant columns contribute nothing
    Z = R ./ s'
    num = 0.0
    den = 0.0
    @inbounds for i in 1:(p - 1)
        zi = view(Z, :, i)
        for j in (i + 1):p
            zj = view(Z, :, j)
            w = zi .* zj
            r = sum(w) / dof
            # variance of the mean of the cross-products
            vr = sum(abs2, w .- mean(w)) * n / (dof^2 * (n - 1))
            num += vr
            den += r^2
        end
    end
    den <= 0 && return 1.0
    return clamp(num / den, 0.0, 1.0)
end

"""
    estimate_total_effects(x, interventions, donors, controls; kwargs...) -> TotalEffectEstimate

Donor-adjusted interventional total effects with standard errors.

For each perturbed gene `u`, all readouts are regressed jointly on
`[intercept, knockout indicator, donor dummies]` over the samples in
`KO(u) ∪ controls`. Writing `Δ` for the knockout-indicator coefficients,
`Θ[u, j] = Δ[j] / Δ[u]`, with the delta-method covariance of that ratio.

# Keyword arguments
- `log_normalize = false`: apply `log1p` first (matches the old `fit_model` path).
- `center = true`, `scale = true`: centre by control means and scale by control
  standard deviations, as in the published implementation. `Θ` is invariant to
  centring; scaling reparameterises `B[i, j]` to `B[i, j] σ_i / σ_j`.
- `covariance = :diagonal`: `:diagonal` keeps per-entry variances only;
  `:full` additionally returns the `p × p` within-knockout covariance.
- `shrink = true`: apply Schäfer–Strimmer shrinkage to the residual covariance
  (only relevant for `covariance = :full`).
- `min_knockdown = 0.25`: rows whose knockdown contrast is smaller than this
  multiple of its own standard error are marked inactive instead of producing
  an unstable ratio.
- `threads = true`: estimate knockout rows in parallel.

`x` is `n × p` expression, `interventions[i, j] == 1` marks sample `i` as a
knockout of gene `j`, `donors` is a per-sample donor code and `controls` flags
the unperturbed samples. This method depends on nothing outside the standard
library, which keeps the stage-1 estimator testable without the modelling stack;
see `graph.jl` for the [`interventionGraph`](@ref) wrapper.
"""
function estimate_total_effects(
        x0::AbstractMatrix{<:Real},
        interventions::AbstractMatrix,
        donors::AbstractVector,
        controls::AbstractVector{Bool};
        targets::Vector{String} = String[],
        log_normalize::Bool = false,
        center::Bool = true,
        scale::Bool = true,
        covariance::Symbol = :diagonal,
        shrink::Bool = true,
        min_knockdown::Float64 = 0.25,
        threads::Bool = true,
        verbose::Bool = true,
    )

    covariance in (:diagonal, :full) ||
        throw(ArgumentError("covariance must be :diagonal or :full, got $covariance"))

    p = size(x0, 2)
    isempty(targets) && (targets = ["gene_$i" for i in 1:p])
    length(targets) == p ||
        throw(ArgumentError("targets has length $(length(targets)) but x has $p columns"))
    x = Matrix{Float64}(x0)
    log_normalize && (x = log1p.(x))

    sum(controls) >= 2 ||
        error("need at least two control samples to form interventional contrasts")

    if center
        x = x .- mean(view(x, controls, :); dims = 1)
    end
    if scale
        sds = vec(std(view(x, controls, :); dims = 1))
        sds[sds .<= 0] .= 1.0
        x = x ./ sds'
    end

    pert = [j for j in 1:p if any(view(interventions, :, j) .== 1)]
    q = length(pert)
    verbose && @info "stage 1: $q perturbed genes, $p readouts, $(sum(controls)) control samples"

    Θ = zeros(q, p)
    se = fill(Inf, q, p)
    # Bool buffers, not BitArrays: BitArray packs 64 entries per word, so
    # threads writing different rows of one column race on the same word.
    active_buf = zeros(Bool, q, p)
    knockdown = zeros(q)
    knockdown_se = fill(NaN, q)
    nobs = zeros(Int, q)
    dof = zeros(Int, q)
    Σ = covariance === :full ? Vector{Matrix{Float64}}(undef, q) : nothing


    function fit_row(k::Int)
        u = pert[k]
        rows = findall((view(interventions, :, u) .== 1) .| collect(controls))
        n = length(rows)
        Y = x[rows, :]
        ko = Float64.(view(interventions, rows, u) .== 1)
        d = donors[rows]
        ud = unique(d)
        # intercept, knockout indicator, donor dummies (first donor = reference)
        D = hcat(ones(n), ko, Float64[d[i] == ud[l] for i in 1:n, l in 2:length(ud)])
        C, resid, rank, F, sinv = _svd_lsq(D, Y)
        df = n - rank
        df < 1 && return (k, false)

        Δ = C[2, :]
        v = _leverage(F, sinv, 2)                 # (DᵀD)⁺[ko, ko]
        ss = vec(sum(abs2, resid; dims = 1)) ./ df
        a = Δ[u]
        se_a = sqrt(max(v * ss[u], 0.0))
        nobs[k] = n
        dof[k] = df
        knockdown[k] = a
        knockdown_se[k] = se_a
        if !(abs(a) > min_knockdown * max(se_a, eps()))
            return (k, false)
        end

        r = Δ ./ a                                 # r[u] == 1 by construction
        ru = view(resid, :, u)
        Euj = (resid' * ru) ./ df                  # E[:, u]
        Euu = Euj[u]
        # delta method for the ratio: Var(r_j) = v/a² (E_jj - 2 r_j E_uj + r_j² E_uu)
        var = (v / a^2) .* (ss .- 2 .* r .* Euj .+ (r .^ 2) .* Euu)
        var[u] = 0.0
        Θ[k, :] = r
        Θ[k, u] = 0.0
        se[k, :] = sqrt.(max.(var, 0.0))
        active_buf[k, :] .= true
        active_buf[k, u] = false

        if covariance === :full
            E = (resid' * resid) ./ df
            if shrink
                λ = schafer_strimmer_shrinkage(resid, df)
                dg = Diagonal(diag(E))
                E = (1 - λ) .* E .+ λ .* dg
            end
            J = Matrix{Float64}(I, p, p)
            @inbounds for j in 1:p
                J[j, u] -= r[j]
            end
            J ./= a
            Σ[k] = Symmetric(J * E * J') |> Matrix
        end
        return (k, true)
    end

    ok = ones(Bool, q)
    if threads && Threads.nthreads() > 1
        Threads.@threads for k in 1:q
            _, good = fit_row(k)
            ok[k] = good
        end
    else
        for k in 1:q
            _, good = fit_row(k)
            ok[k] = good
        end
    end

    nbad = count(!, ok)
    if nbad > 0 && verbose
        @warn "$nbad knockout(s) had no identifiable knockdown and were dropped from the likelihood"
    end

    weak = [k for k in 1:q if ok[k] && abs(knockdown[k]) < sqrt(10) * knockdown_se[k]]
    if !isempty(weak) && verbose
        @warn "weak knockdown (first-stage F < 10) for $(length(weak)) perturbed gene(s): $(targets[pert[weak]]). Their total effects have heavy-tailed errors; they are down-weighted in fit_llcb but will distort the closed-form estimate."
    end
    return TotalEffectEstimate(Θ, se, Σ, knockdown, knockdown_se, BitMatrix(active_buf), nobs, dof, pert, targets)
end

