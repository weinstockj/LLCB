
# =============================================================================
# Stage 2 of LLCB: the exact interventional moment model.
#
# For a linear structural system x = Bᵀx + e under a *hard* intervention on
# node u, the normalised total-effect matrix is
#
#       T̃[u, j] = A[u, j] / A[u, u],        A = (I - B)⁻¹ .
#
# Proof. Row u of A(I - B) = eᵤᵀ gives A[u, v] = Σ_w A[u, w] B[w, v] for v ≠ u.
# Dividing by A[u, u] yields h[v] = Σ_w h[w] B[w, v] with h[u] = 1, which is
# exactly the recursion "total effect on v = Σ over parents of (total effect on
# the parent) × (direct effect)" that LLCB encodes. The solution is unique
# whenever I - B[-u, -u] is invertible, so h = T̃[u, ·].
#
# Two consequences worth stating. First, T̃ ≠ (I - B)⁻¹ unless the graph is
# acyclic: a hard intervention cuts the edges *into* u, so feedback loops
# through u must be removed, and the row normalisation by A[u, u] is exactly
# that removal. Second, the entire matrix follows from ONE p × p inverse, not
# from p separate (p-1) × (p-1) inverses.
#
# Writing the moment condition this way -- model on the left, data only on the
# right -- is what removes the errors-in-variables problem of the original
# formulation t = T β, in which the noisy total effects appear as a design
# matrix and attenuate β. It also removes the dense
# nv(nv-1) × nv(nv-1) matrix the original code allocated, which is the hard
# memory wall of that implementation.
# =============================================================================

const _PRIORS = (:normal, :studentt, :horseshoe)

"""
    LLCBModel

Log-posterior for the structural coefficient matrix `B` given stage-1 total
effects, with an analytic gradient (no automatic differentiation). The
`LogDensityProblems` interface is attached in `fit_llcb.jl`, which keeps this
file free of dependencies outside the standard library.

Construct with [`llcb_model`](@ref).
"""
struct LLCBModel
    Θ::Matrix{Float64}          # q × p observed total effects
    W::Matrix{Float64}          # q × p precision weights, 0 where inactive
    rows::Vector{Int}           # perturbed node index of each row of Θ
    p::Int
    q::Int
    nactive::Int
    prior::Symbol
    prior_scale::Float64        # scale for :normal / :studentt
    prior_df::Float64           # degrees of freedom for :studentt
    tau0::Float64               # global-scale prior for :horseshoe
    slab_scale::Float64
    slab_df::Float64
    rho_max::Float64            # stability target for the largest singular value
    stab_scale::Float64         # softness of the stability penalty
    estimate_dispersion::Bool
    m::Int                      # number of free entries of B = p(p-1)
    dim::Int
end

"""
    llcb_model(te::TotalEffectEstimate; kwargs...) -> LLCBModel

Build the stage-2 log-posterior from stage-1 output.

Weights are the reciprocal stage-1 sampling variances, so the free per-block
`σ` of the original model is replaced by uncertainty that was actually
estimated. An optional dispersion `φ` can rescale all of them, but it is off by
default because the LLCB design leaves no residual degrees of freedom to
identify it (see `estimate_dispersion`).

# Keyword arguments
- `prior = :studentt`: one of `:normal`, `:studentt`, `:horseshoe`.
  `:studentt` keeps the parameter count at `p(p-1) + 2` and is the scalable
  default; `:horseshoe` is the regularised horseshoe of Piironen & Vehtari
  (2017) in non-centred form, which adds `p(p-1) + 2` auxiliary parameters.
- `prior_scale = nothing`, `prior_df = 3.0`: scale and degrees of freedom for
  `:normal` / `:studentt`. When `nothing`, the scale is set to the median
  absolute observed total effect, which puts the prior on the same scale as the
  coefficients; a scale far below the true magnitudes pushes the posterior into
  the prior's tail and degrades sampling.
- `tau0 = 0.01`, `slab_scale = 0.5`, `slab_df = 4.0`: horseshoe hyperparameters.
- `rho_max = 2.0`, `stab_scale = 0.5`: a smooth penalty on the largest singular
  value of `B`, acting only as a guard against divergence. It is deliberately
  loose: `σ_max(B) ≥ ρ(B)`, often by a wide margin, so a tight bound on the
  singular value would forbid perfectly legitimate cyclic networks and, being
  stiff, would collapse the sampler's step size. The hard guard against a
  non-invertible `I - B` is separate and always on. This replaces the original
  `spectral_radius` penalty, which seeded power iteration with `Base.rand`
  *inside* the model -- making the log density stochastic and its gradient
  irreproducible -- and which in any case returned the spectral radius of the
  elementwise square `W ∘ W` rather than of `W`.
- `estimate_dispersion = nothing`: whether to sample `log φ`. The default turns
  it on only when there are residual degrees of freedom (more weighted moments
  than coefficients), which never happens in the standard LLCB design; see the
  comment in the function body. `false` fixes `φ ≡ 1`.
- `weight_floor = 1e-8`: guards against division by a vanishing standard error.
"""
function llcb_model(
        te::TotalEffectEstimate;
        prior::Symbol = :studentt,
        prior_scale::Union{Nothing,Float64} = nothing,
        prior_df::Float64 = 3.0,
        tau0::Float64 = 0.01,
        slab_scale::Float64 = 0.5,
        slab_df::Float64 = 4.0,
        rho_max::Float64 = 2.0,
        stab_scale::Float64 = 0.5,
        estimate_dispersion::Union{Nothing,Bool} = nothing,
        weight_floor::Float64 = 1e-8,
    )
    prior in _PRIORS || throw(ArgumentError("prior must be one of $(_PRIORS), got $prior"))
    q, p = size(te.Θ)
    # A prior scale far below the true coefficient magnitudes forces the
    # posterior deep into the prior's tail, where curvature changes quickly and
    # the sampler struggles. Default it to the typical size of the observed
    # total effects, which share the scale of the direct effects.
    if prior_scale === nothing
        vals = [abs(te.Θ[k, j]) for j in 1:p, k in 1:q if te.active[k, j]]
        prior_scale = isempty(vals) ? 0.1 : max(0.05, median(vals))
    end
    W = zeros(q, p)
    @inbounds for j in 1:p, k in 1:q
        if te.active[k, j] && isfinite(te.se[k, j]) && te.se[k, j] > weight_floor
            W[k, j] = inv(te.se[k, j]^2)
        end
    end
    nactive = count(>(0.0), W)
    nactive > 0 || error("no usable stage-1 total effects: all weights are zero")
    m = p * (p - 1)
    # A dispersion φ is identified only by residual degrees of freedom, and
    # there are none: q perturbed genes give q(p - 1) <= p(p - 1) moments for
    # p(p - 1) coefficients. With every readout perturbed the fit is exact at
    # the closed-form point, so the marginal of φ is its prior times φ^(1/2)
    # and the joint posterior is a funnel (B collapses as φ -> 0) that NUTS
    # cannot traverse. Default: φ ≡ 1, i.e. trust the stage-1 standard errors.
    resid_dof = nactive - m
    if estimate_dispersion === nothing
        estimate_dispersion = resid_dof > 0
    elseif estimate_dispersion && resid_dof <= 0
        @warn "estimate_dispersion = true with $resid_dof residual degrees of freedom: φ is not identified by the data and induces a funnel in the posterior"
    end
    # :normal / :studentt add no auxiliary parameters; :horseshoe adds log λ
    # (m of them) plus log τ and log c.
    nextra = (prior === :horseshoe ? m + 2 : 0) + (estimate_dispersion ? 1 : 0)
    return LLCBModel(
        Matrix{Float64}(te.Θ), W, copy(te.perturbed), p, q, nactive,
        prior, prior_scale, prior_df, tau0, slab_scale, slab_df,
        rho_max, stab_scale, estimate_dispersion, m, m + nextra,
    )
end

"""
    llcb_dimension(model::LLCBModel)

Number of sampled parameters.
"""
llcb_dimension(model::LLCBModel) = model.dim

# ----------------------------------------------------------------------------
# packing: free entries of B are the off-diagonal ones, column-major
# ----------------------------------------------------------------------------

"""
    free_indices(p) -> Vector{Int}

Linear indices of the off-diagonal entries of a `p × p` matrix, column-major.
`B[i, j]` is the direct effect of parent `i` on child `j`, and `B[j, j] ≡ 0`.
"""
function free_indices(p::Int)
    idx = Vector{Int}(undef, p * (p - 1))
    c = 0
    @inbounds for j in 1:p, i in 1:p
        i == j && continue
        c += 1
        idx[c] = (j - 1) * p + i
    end
    return idx
end

"""
    to_matrix(b::AbstractVector, p) -> Matrix

Scatter the free coefficients into a `p × p` matrix with zero diagonal.
"""
function to_matrix(b::AbstractVector{T}, p::Int) where {T}
    B = zeros(T, p, p)
    c = 0
    @inbounds for j in 1:p, i in 1:p
        i == j && continue
        c += 1
        B[i, j] = b[c]
    end
    return B
end

"""
    from_matrix(B::AbstractMatrix) -> Vector

Gather the off-diagonal entries of `B` in the packing used by [`to_matrix`](@ref).
"""
function from_matrix(B::AbstractMatrix{T}) where {T}
    p = size(B, 1)
    b = Vector{T}(undef, p * (p - 1))
    c = 0
    @inbounds for j in 1:p, i in 1:p
        i == j && continue
        c += 1
        b[c] = B[i, j]
    end
    return b
end

# ----------------------------------------------------------------------------
# implied total effects and the data-fit term
# ----------------------------------------------------------------------------

"""
    implied_total_effects(B) -> Matrix

The hard-intervention total-effect matrix implied by direct effects `B`:
`T̃ = Diagonal(diag(A))⁻¹ A` with `A = (I - B)⁻¹`, so `T̃[u, u] = 1`.
Returns `T̃` with the self-effects left at 1; subtract `I` for the convention
used by [`TotalEffectEstimate`](@ref).
"""
function implied_total_effects(B::AbstractMatrix{Float64})
    p = size(B, 1)
    A = inv(Matrix{Float64}(I, p, p) - B)
    return A ./ diag(A)
end

# dominant singular value by deterministic power iteration. Deterministic is
# the point: the original implementation drew a random start vector inside the
# model, so repeated evaluations of the same parameters gave different log
# densities. dσ/dB = u vᵀ exactly, with u, v the dominant singular vectors.
function _sigma_max(B::Matrix{Float64}; iters::Int = 32, tol::Float64 = 1e-10)
    p = size(B, 2)
    v = fill(1.0 / sqrt(p), p)
    u = similar(v, size(B, 1))
    σ = 0.0
    @inbounds for _ in 1:iters
        mul!(u, B, v)
        nu = norm(u)
        nu <= 0 && return 0.0, zeros(size(B, 1)), v
        u ./= nu
        mul!(v, B', u)
        nv = norm(v)
        nv <= 0 && return 0.0, u, zeros(p)
        v ./= nv
        if abs(nv - σ) < tol * max(nv, 1.0)
            σ = nv
            break
        end
        σ = nv
    end
    return σ, u, v
end

"""
    data_term!(grad_B, model, B) -> Float64

Weighted sum of squares between the implied and observed total effects, and
its gradient with respect to `B`.

The gradient is analytic. With `A = (I - B)⁻¹`, `d = diag(A)`, `S = A ./ d`,
residual `R = S[rows, :] - I_rows - Θ` and `G = W ∘ R` scattered back to
`p × p`, the chain rule through the row normalisation and the matrix inverse
gives

    ∂f/∂A = Diagonal(d)⁻¹ G - Diagonal(g),   g[u] = (1/d[u]) Σ_j G[u, j] S[u, j]
    ∂f/∂B = Aᵀ (∂f/∂A) Aᵀ

at a cost of one factorisation and two matrix products, independent of how
many coefficients are being sampled.
"""
function data_term!(gB::Matrix{Float64}, model::LLCBModel, B::Matrix{Float64})
    p = model.p
    M = Matrix{Float64}(I, p, p)
    M .-= B
    local A::Matrix{Float64}
    try
        A = inv(M)
    catch
        return Inf
    end
    (all(isfinite, A) && maximum(abs, A) < 1e10) || return Inf
    d = diag(A)
    any(x -> abs(x) < 1e-10, d) && return Inf

    S = A ./ d                                   # row u divided by A[u, u]
    Gfull = zeros(p, p)
    f = 0.0
    @inbounds for k in 1:model.q
        u = model.rows[k]
        for j in 1:p
            w = model.W[k, j]
            w == 0.0 && continue
            r = S[u, j] - (j == u ? 1.0 : 0.0) - model.Θ[k, j]
            f += 0.5 * w * r * r
            Gfull[u, j] = w * r
        end
    end

    gvec = zeros(p)
    @inbounds for u in 1:p
        acc = 0.0
        for j in 1:p
            acc += Gfull[u, j] * S[u, j]
        end
        gvec[u] = acc / d[u]
    end
    Gbar = Gfull ./ d
    @inbounds for u in 1:p
        Gbar[u, u] -= gvec[u]
    end
    mul!(gB, A', Gbar * A')
    @inbounds for i in 1:p
        gB[i, i] = 0.0
    end
    return f
end

# ----------------------------------------------------------------------------
# log posterior
# ----------------------------------------------------------------------------

"""
    llcb_logdensity_and_gradient(model::LLCBModel, θ) -> (logp, grad)

Analytic log posterior and gradient. The parameter vector is

- `:normal` / `:studentt`: `[vec(B); log φ]`
- `:horseshoe`: `[z; log λ; log τ; log c; log φ]` with
  `B = z τ λ̃`, `λ̃ = c λ / sqrt(c² + τ² λ²)`

and `log φ` is omitted when `estimate_dispersion = false`.
"""
function llcb_logdensity_and_gradient(model::LLCBModel, θ::AbstractVector{Float64})
    p, m = model.p, model.m
    all(isfinite, θ) || return (-Inf, zeros(length(θ)))
    g = zeros(length(θ))

    # --- unpack ------------------------------------------------------------
    local b::Vector{Float64}
    if model.prior === :horseshoe
        z = view(θ, 1:m)
        logλ = view(θ, (m + 1):(2m))
        logτ = θ[2m + 1]
        logc = θ[2m + 2]
        λ = exp.(logλ); τ = exp(logτ); c = exp(logc)
        denom = sqrt.(c^2 .+ (τ^2) .* (λ .^ 2))
        λ̃ = (c .* λ) ./ denom
        b = z .* (τ .* λ̃)
        iφ = model.estimate_dispersion ? 2m + 3 : 0
    else
        b = Vector{Float64}(view(θ, 1:m))
        iφ = model.estimate_dispersion ? m + 1 : 0
    end
    logφ = model.estimate_dispersion ? θ[iφ] : 0.0
    φ = exp(logφ)

    B = to_matrix(b, p)

    # --- data term ---------------------------------------------------------
    gB = zeros(p, p)
    sse = data_term!(gB, model, B)      # 0.5 Σ w r²  (unscaled by φ)
    isfinite(sse) || return (-Inf, g)

    logp = -sse / φ - 0.5 * model.nactive * logφ
    gb = from_matrix(gB) ./ (-φ)        # ∂logp/∂b
    if model.estimate_dispersion
        g[iφ] = sse / φ - 0.5 * model.nactive
        # weak half-Cauchy(0, 1) on sqrt(φ): keeps φ from running away
        logp += -log1p(φ) + 0.5 * logφ
        g[iφ] += -φ / (1 + φ) + 0.5
    end

    # --- stability penalty -------------------------------------------------
    σ, uvec, vvec = _sigma_max(B)
    if σ > model.rho_max
        e = (σ - model.rho_max) / model.stab_scale
        logp -= 0.5 * e^2
        coef = -e / model.stab_scale
        @inbounds for j in 1:p, i in 1:p
            i == j && continue
            gB[i, j] = uvec[i] * vvec[j]          # ∂σ/∂B
        end
        gb .+= coef .* from_matrix(gB)
    end

    # --- prior -------------------------------------------------------------
    if model.prior === :normal
        s2 = model.prior_scale^2
        logp += -0.5 * sum(abs2, b) / s2
        gb .+= -b ./ s2
        g[1:m] .= gb
    elseif model.prior === :studentt
        ν, s = model.prior_df, model.prior_scale
        @inbounds for i in 1:m
            t = b[i] / s
            logp += -0.5 * (ν + 1) * log1p(t^2 / ν)
            gb[i] += -(ν + 1) * b[i] / (ν * s^2 + b[i]^2)
        end
        g[1:m] .= gb
    else # :horseshoe
        z = view(θ, 1:m)
        logλ = view(θ, (m + 1):(2m))
        logτ = θ[2m + 1]; logc = θ[2m + 2]
        λ = exp.(logλ); τ = exp(logτ); c = exp(logc)
        s2 = c^2 .+ (τ^2) .* (λ .^ 2)
        denom = sqrt.(s2)
        λ̃ = (c .* λ) ./ denom
        # b = z τ λ̃
        dλ̃_dλ = (c^3) ./ (s2 .* denom)
        dλ̃_dτ = -(c .* (λ .^ 3) .* τ) ./ (s2 .* denom)
        dλ̃_dc = ((λ .^ 3) .* τ^2) ./ (s2 .* denom)
        # chain the data+stability gradient through the reparameterisation
        @inbounds for i in 1:m
            g[i] += gb[i] * τ * λ̃[i]                              # ∂/∂z
            g[m + i] += gb[i] * z[i] * τ * dλ̃_dλ[i] * λ[i]        # ∂/∂log λ
            g[2m + 1] += gb[i] * z[i] * (λ̃[i] + τ * dλ̃_dτ[i]) * τ # ∂/∂log τ
            g[2m + 2] += gb[i] * z[i] * τ * dλ̃_dc[i] * c          # ∂/∂log c
        end
        # z ~ Normal(0, 1)
        logp += -0.5 * sum(abs2, z)
        @inbounds for i in 1:m
            g[i] -= z[i]
        end
        # λ ~ Cauchy⁺(0, 1), sampled as log λ
        @inbounds for i in 1:m
            logp += -log1p(λ[i]^2) + logλ[i]
            g[m + i] += 1 - 2 * λ[i]^2 / (1 + λ[i]^2)
        end
        # τ ~ Cauchy⁺(0, τ0), sampled as log τ
        logp += -log1p((τ / model.tau0)^2) + logτ
        g[2m + 1] += 1 - 2 * (τ / model.tau0)^2 / (1 + (τ / model.tau0)^2)
        # c² ~ InvGamma(ν/2, ν s²/2), sampled as log c
        ν, s = model.slab_df, model.slab_scale
        c2 = c^2
        logp += -(ν / 2 + 1) * log(c2) - (ν * s^2 / 2) / c2 + 2 * logc
        # d/dlog c: -(ν+2) from the density, +2 from the log-scale Jacobian
        g[2m + 2] += -ν + (ν * s^2) / c2
    end

    isfinite(logp) || return (-Inf, zeros(length(θ)))
    return (logp, g)
end

llcb_logdensity(model::LLCBModel, θ::AbstractVector{Float64}) =
    first(llcb_logdensity_and_gradient(model, θ))

"""
    dispersion_index(model::LLCBModel) -> Int

Index of `log φ` in the parameter vector, or `0` when the dispersion is fixed.
"""
dispersion_index(model::LLCBModel) =
    model.estimate_dispersion ?
        (model.prior === :horseshoe ? 2 * model.m + 3 : model.m + 1) : 0

"""
    coefficients(model::LLCBModel, θ) -> Matrix

Map a parameter vector back to the `p × p` direct-effect matrix `B`.
"""
function coefficients(model::LLCBModel, θ::AbstractVector{Float64})
    m = model.m
    if model.prior === :horseshoe
        z = view(θ, 1:m)
        λ = exp.(view(θ, (m + 1):(2m)))
        τ = exp(θ[2m + 1]); c = exp(θ[2m + 2])
        λ̃ = (c .* λ) ./ sqrt.(c^2 .+ (τ^2) .* (λ .^ 2))
        return to_matrix(z .* (τ .* λ̃), model.p)
    else
        return to_matrix(view(θ, 1:m), model.p)
    end
end

"""
    initial_parameters(model::LLCBModel; B0 = nothing, rng) -> Vector

A starting point for optimisation or sampling, optionally seeded with a
direct-effect matrix `B0`.
"""
function initial_parameters(model::LLCBModel; B0 = nothing, rng = Random.GLOBAL_RNG)
    m = model.m
    θ = zeros(model.dim)
    b = B0 === nothing ? 1e-3 .* randn(rng, m) : from_matrix(B0)
    if model.prior === :horseshoe
        τ0 = model.tau0
        θ[1:m] .= b ./ τ0
        θ[(m + 1):(2m)] .= 0.0
        θ[2m + 1] = log(τ0)
        θ[2m + 2] = log(model.slab_scale)
    else
        θ[1:m] .= b
    end
    model.estimate_dispersion && (θ[end] = 0.0)
    return θ
end


# ----------------------------------------------------------------------------
# closed-form inversion
# ----------------------------------------------------------------------------

"""
    closed_form_coefficients(T̃) -> Matrix

Invert the moment identity exactly. With `T̃ = Diagonal(d)⁻¹ A` and
`A = (I - B)⁻¹`, `I - B = T̃⁻¹ Diagonal(d)⁻¹`; the zero diagonal of `B` forces
`d[j] = (T̃⁻¹)[j, j]`, so

    B = I - T̃⁻¹ Diagonal(diag(T̃⁻¹))⁻¹ .

`T̃` must be the full `p × p` normalised total-effect matrix with unit
diagonal, i.e. every readout must also have been perturbed. With noise-free
total effects this recovers `B` exactly; with estimated ones it is the
method-of-moments estimator, which is where the MAP converges under a flat
prior because the system is just-identified. It provides no sparsity, and it
amplifies noise when `T̃` is ill-conditioned.
"""
function closed_form_coefficients(T̃::AbstractMatrix{<:Real})
    p = size(T̃, 1)
    size(T̃, 2) == p || throw(DimensionMismatch("total-effect matrix must be square, got $(size(T̃))"))
    F = lu(Matrix{Float64}(T̃); check = false)
    issuccess(F) || throw(SingularException(0))
    Ti = inv(F)
    dg = diag(Ti)
    any(x -> abs(x) < 1e-12, dg) && throw(SingularException(0))
    B = Matrix{Float64}(I, p, p) .- Ti ./ dg'
    @inbounds for j in 1:p
        B[j, j] = 0.0
    end
    return B
end

"""
    closed_form_coefficients(te::TotalEffectEstimate) -> Matrix

Closed-form direct effects from stage-1 output. Requires every readout to have
been perturbed with an identifiable knockdown, so that the full `p × p`
total-effect matrix is available.
"""
function closed_form_coefficients(te::TotalEffectEstimate)
    q, p = size(te.Θ)
    full = q == p && te.perturbed == collect(1:p) &&
           all(te.active[k, j] for k in 1:p, j in 1:p if k != j)
    full || throw(ArgumentError(
        "closed-form inversion needs every readout perturbed with an identifiable " *
        "knockdown (got $q perturbed of $p readouts, $(count(te.active)) of $(p * (p - 1)) active moments)"))
    weak = weak_knockdowns(te)
    isempty(weak) || @warn "closed-form inversion weights every moment equally; $(length(weak)) row(s) have weak knockdowns ($(te.targets[te.perturbed[weak]])) and will distort it. Prefer fit_llcb, which weights by the stage-1 standard errors."
    return closed_form_coefficients(te.Θ + Matrix{Float64}(I, p, p))
end

"""
    has_full_total_effects(te::TotalEffectEstimate) -> Bool

Whether [`closed_form_coefficients`](@ref) applies to `te`.
"""
function has_full_total_effects(te::TotalEffectEstimate)
    q, p = size(te.Θ)
    return q == p && te.perturbed == collect(1:p) &&
           all(te.active[k, j] for k in 1:p, j in 1:p if k != j)
end
