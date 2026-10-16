
# =============================================================================
# Inference for the stage-2 moment model: MAP, Pathfinder preconditioning,
# then NUTS driven by the analytic gradient.
#
# The published pipeline reported Pathfinder's Gaussian approximation as the
# posterior. Here Pathfinder provides a warm start and the HMC metric, and the
# reported posterior comes from NUTS, which gives better-calibrated intervals in
# simulation.
# =============================================================================

# The LogDensityProblems interface lives here rather than in moment_model.jl so
# that the numerical core (total effects, implied total effects, the analytic
# gradient) depends on nothing outside the standard library and can be tested
# without loading the sampling stack.
LogDensityProblems.dimension(model::LLCBModel) = llcb_dimension(model)
LogDensityProblems.capabilities(::Type{LLCBModel}) =
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(model::LLCBModel, θ::AbstractVector{Float64}) =
    llcb_logdensity(model, θ)
LogDensityProblems.logdensity_and_gradient(model::LLCBModel, θ::AbstractVector{Float64}) =
    llcb_logdensity_and_gradient(model, θ)

"""
    LLCBFit

Posterior summaries for the direct-effect matrix `B`.

# Fields
- `B_mean`, `B_sd`: posterior mean and standard deviation, `p × p`.
- `prob_pos`: posterior probability that an edge is positive.
- `lsfr`: local sign false rate, `min(P(B > 0), P(B < 0))` -- the threshold-free
  evidence that an edge exists with a definite sign.
- `pip`: `P(|B| > pip_threshold)`, kept for comparability with the published
  output, which used a fixed `0.05` cut.
- `quantiles`: posterior quantiles, keyed by probability.
- `rhat`, `ess`: per-edge convergence diagnostics.
- `map_B`: the MAP estimate.
- `draws`: thinned post-warmup draws of the free coefficients,
  `(ndraws, m, nchains)`, stored in single precision.
- `diagnostics`: step sizes, divergence counts, acceptance, timings, the
  dispersion `φ`, which metric was used, and the spectral radius of the
  posterior-mean `B`.
"""
struct LLCBFit
    B_mean::Matrix{Float64}
    B_sd::Matrix{Float64}
    prob_pos::Matrix{Float64}
    lsfr::Matrix{Float64}
    pip::Matrix{Float64}
    quantiles::Dict{Float64,Matrix{Float64}}
    rhat::Matrix{Float64}
    ess::Matrix{Float64}
    map_B::Matrix{Float64}
    draws::Array{Float32,3}
    diagnostics::Dict{Symbol,Any}
    targets::Vector{String}
    perturbed::Vector{Int}
    pip_threshold::Float64
end

"""
    map_estimate(model::LLCBModel; θ0, iterations, show_trace) -> (θ, result)

MAP by L-BFGS on the analytic gradient.
"""
function map_estimate(
        model::LLCBModel;
        θ0::Union{Nothing,Vector{Float64}} = nothing,
        iterations::Int = 2_000,
        show_trace::Bool = false,
        g_tol::Float64 = 1e-8,
    )
    θstart = θ0 === nothing ? initial_parameters(model) : copy(θ0)
    f(θ) = -llcb_logdensity(model, θ)
    function g!(G, θ)
        _, grad = llcb_logdensity_and_gradient(model, θ)
        G .= .-grad
        return G
    end
    function fg!(G, θ)
        lp, grad = llcb_logdensity_and_gradient(model, θ)
        G .= .-grad
        return -lp
    end
    # Optim 2 no longer re-exports `only_fg!`; build the objective explicitly.
    obj = Optim.OnceDifferentiable(f, g!, fg!, θstart)
    res = Optim.optimize(
        obj, θstart,
        Optim.LBFGS(; m = 10, linesearch = LineSearches.BackTracking()),
        Optim.Options(; iterations = iterations, show_trace = show_trace, g_abstol = g_tol),
    )
    return Optim.minimizer(res), res
end

# One NUTS chain against the analytic gradient.
#
# The moment likelihood is a weighted least-squares surface pushed through a
# matrix inverse, so its curvature is strongly anisotropic and highly
# correlated across coefficients. A diagonal metric cannot represent that: step
# size adaptation collapses toward zero and the chains stop moving (in testing,
# step size ~1e-4 with rhat ~3 and ESS ~5). When Pathfinder has produced a
# usable covariance we therefore use it as a fixed metric -- Pathfinder's
# diagonal-plus-low-rank `RankUpdateEuclideanMetric`, which stays affordable at
# large p -- and adapt only the step size.
function _nuts_chain(
        model::LLCBModel, θ0::Vector{Float64}, n_adapt::Int, n_samples::Int;
        rng, target_accept::Float64, max_depth::Int, metric, adapt_mass::Bool,
    )
    ℓπ = θ -> llcb_logdensity(model, θ)
    ∂ℓπ = θ -> llcb_logdensity_and_gradient(model, θ)
    h = AdvancedHMC.Hamiltonian(metric, ℓπ, ∂ℓπ)
    ϵ = AdvancedHMC.find_good_stepsize(rng, h, θ0)
    integrator = AdvancedHMC.Leapfrog(ϵ)
    kernel = AdvancedHMC.HMCKernel(
        AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(
            integrator, AdvancedHMC.GeneralisedNoUTurn(; max_depth = max_depth),
        ),
    )
    adaptor = adapt_mass ?
        AdvancedHMC.StanHMCAdaptor(
            AdvancedHMC.MassMatrixAdaptor(metric),
            AdvancedHMC.StepSizeAdaptor(target_accept, integrator),
        ) :
        AdvancedHMC.StepSizeAdaptor(target_accept, integrator)
    samples, stats = AdvancedHMC.sample(
        rng, h, kernel, θ0, n_samples + n_adapt, adaptor, n_adapt;
        drop_warmup = true, progress = false, verbose = false,
    )
    return samples, stats
end

# rank-normalised split-Rhat and bulk ESS for one parameter.
function _ess_rhat(x::AbstractMatrix{Float64})
    n, c = size(x)
    (n < 8 || c < 2) && return (Float64(n * c), NaN)
    try
        arr = Array{Float64,3}(undef, n, c, 1)
        arr[:, :, 1] .= x
        e, r = MCMCDiagnosticTools.ess_rhat(arr)
        return (Float64(first(e)), Float64(first(r)))
    catch
        return (NaN, NaN)
    end
end

"""
    fit_llcb(te::TotalEffectEstimate; kwargs...) -> (fit::LLCBFit, model::LLCBModel)

Fit the LLCB direct-effect matrix from stage-1 total effects.

Runs MAP by L-BFGS, preconditions with Pathfinder, then samples `nchains` NUTS
chains in parallel across Julia threads.

# Keyword arguments
- `prior`, `prior_scale`, `prior_df`, `rho_max`, ... : forwarded to
  [`llcb_model`](@ref) when `model` is not supplied.
- `model`: a prebuilt [`LLCBModel`](@ref), bypassing construction.
- `nchains = max(2, min(4, Threads.nthreads()))`, `n_adapt = 800`,
  `n_samples = 800`, `target_accept = 0.9`, `max_depth = 10`.
- `use_pathfinder = true`, `pathfinder_runs = 4`, `pathfinder_jitter = 0.5`:
  run Pathfinder from dispersed starts around the MAP and use the covariance of
  its best path (by ELBO) as the HMC metric. Starting Pathfinder *at* the MAP
  gives a degenerate curvature estimate and an unreliable metric.
- `retry_on_poor_mixing = true`, `rhat_tolerance = 1.05`,
  `divergence_tolerance = 0.02`: if the Pathfinder metric yields chains that
  fail these thresholds, the sampling pass is repeated once with a diagonal
  metric seeded from the Pathfinder variances and full Stan-style adaptation,
  and whichever pass mixed better is reported.
- `blas_threads = :auto`: BLAS threads during sampling; `:auto` uses one
  thread when chains run in parallel, which avoids oversubscribing cores.
- `max_stored = 500`: draws are thinned so that at most this many per chain are
  kept for quantiles and diagnostics; means, standard deviations, sign
  probabilities and PIPs are accumulated online over every draw.
- `pip_threshold = 0.05`: the effect-size cut used for `pip`.
"""
function fit_llcb(
        te::TotalEffectEstimate;
        model::Union{Nothing,LLCBModel} = nothing,
        nchains::Int = max(2, min(4, Threads.nthreads())),
        n_adapt::Int = 800,
        n_samples::Int = 800,
        target_accept::Float64 = 0.9,
        max_depth::Int = 10,
        use_pathfinder::Bool = true,
        pathfinder_runs::Int = 4,
        pathfinder_jitter::Float64 = 0.5,
        retry_on_poor_mixing::Bool = true,
        rhat_tolerance::Float64 = 1.05,
        divergence_tolerance::Float64 = 0.02,
        max_stored::Int = 500,
        blas_threads::Union{Symbol,Int} = :auto,
        pip_threshold::Float64 = 0.05,
        quantile_probs::Vector{Float64} = [0.025, 0.5, 0.975],
        seed::Int = 1,
        verbose::Bool = true,
        kwargs...,
    )
    mdl = model === nothing ? llcb_model(te; kwargs...) : model
    p, m = mdl.p, mdl.m
    verbose && @info "$(now()) stage 2: $(p) nodes, $(m) free coefficients, $(mdl.nactive) weighted moments, prior = $(mdl.prior)"

    # Start the MAP from the closed-form inversion when every readout was
    # perturbed: the system is just-identified, so that point is already the
    # flat-prior optimum and L-BFGS only has to account for the prior.
    map_start = :default
    θ0 = nothing
    if has_full_total_effects(te)
        try
            θ0 = initial_parameters(mdl; B0 = closed_form_coefficients(te))
            map_start = :closed_form
        catch err
            verbose && @warn "closed-form start unavailable; using the default start" exception = err
        end
    end
    t0 = time()
    θmap, optres = map_estimate(mdl; θ0 = θ0)
    t_map = time() - t0
    verbose && @info "$(now()) MAP in $(round(t_map; digits = 2))s, logp = $(round(llcb_logdensity(mdl, θmap); digits = 2))"

    # --- Pathfinder preconditioning ---------------------------------------
    θstart = copy(θmap)
    pf_result = nothing
    pf_metric = nothing
    pf_var = nothing
    t_pf = 0.0
    if use_pathfinder
        try
            t1 = time()
            rng_pf = Random.MersenneTwister(seed)
            inits = [θmap .+ pathfinder_jitter .* randn(rng_pf, length(θmap))
                     for _ in 1:pathfinder_runs]
            pf_result = Pathfinder.multipathfinder(
                mdl, 200; nruns = pathfinder_runs, init = inits, rng = rng_pf,
            )
            elbos = [try
                         v = pf_result.pathfinder_results[i].elbo_estimates[end].value
                         isfinite(v) ? v : -Inf
                     catch
                         -Inf
                     end for i in eachindex(pf_result.pathfinder_results)]
            best_path = argmax(elbos)
            Σ = pf_result.pathfinder_results[best_path].fit_distribution.Σ
            pf_metric = AdvancedHMC.RankUpdateEuclideanMetric(Σ)   # moved from Pathfinder to AdvancedHMC
            pf_var = max.(diag(Σ), 1e-12)
            cand = pf_result.draws
            best = argmax([llcb_logdensity(mdl, Vector{Float64}(view(cand, :, i)))
                           for i in axes(cand, 2)])
            θcand = Vector{Float64}(view(cand, :, best))
            llcb_logdensity(mdl, θcand) > llcb_logdensity(mdl, θstart) && (θstart = θcand)
            t_pf = time() - t1
            verbose && @info "$(now()) multipathfinder ($pathfinder_runs runs) in $(round(t_pf; digits = 2))s, best ELBO $(round(elbos[best_path]; digits = 2))"
        catch err
            @warn "pathfinder failed; falling back to the MAP start and a diagonal metric" exception = err
            pf_result = nothing; pf_metric = nothing; pf_var = nothing
        end
    end

    # --- one sampling pass -------------------------------------------------
    thin = max(1, cld(n_samples, max_stored))
    nstored = length(1:thin:n_samples)

    function sampling_pass(metric_for, adapt_mass::Bool, tag::Symbol)
        draws = Array{Float32,3}(undef, nstored, m, nchains)
        sum_b = zeros(m, nchains); sumsq_b = zeros(m, nchains)
        npos = zeros(Int, m, nchains); npip = zeros(Int, m, nchains)
        φ_chain = zeros(nchains); eps_chain = zeros(nchains)
        ndiv = zeros(Int, nchains); acc_chain = zeros(nchains)
        iφ = dispersion_index(mdl)
        t2 = time()
        Threads.@threads for c in 1:nchains
            rng = Random.MersenneTwister(seed + 1000 * c + (tag === :diagonal ? 77 : 0))
            # `local`: θ0 also names a variable of fit_llcb, and an unqualified
            # assignment here would rebind that shared variable from every thread
            local θ0 = copy(θstart)
            c > 1 && (θ0 .+= 0.01 .* randn(rng, length(θ0)))
            samples, stats = _nuts_chain(
                mdl, θ0, n_adapt, n_samples;
                rng = rng, target_accept = target_accept, max_depth = max_depth,
                metric = metric_for, adapt_mass = adapt_mass,
            )
            store = 0; φacc = 0.0
            @inbounds for (s, θ) in enumerate(samples)
                b = from_matrix(coefficients(mdl, θ))
                for i in 1:m
                    sum_b[i, c] += b[i]; sumsq_b[i, c] += b[i]^2
                    b[i] > 0 && (npos[i, c] += 1)
                    abs(b[i]) > pip_threshold && (npip[i, c] += 1)
                end
                iφ > 0 && (φacc += exp(θ[iφ]))
                if (s - 1) % thin == 0 && store < nstored
                    store += 1
                    @views draws[store, :, c] .= Float32.(b)
                end
            end
            φ_chain[c] = iφ > 0 ? φacc / length(samples) : 1.0
            eps_chain[c] = mean(getproperty.(stats, :step_size))
            ndiv[c] = count(getproperty.(stats, :numerical_error))
            acc_chain[c] = mean(getproperty.(stats, :acceptance_rate))
        end
        seconds = time() - t2
        rh = Vector{Float64}(undef, m); es = Vector{Float64}(undef, m)
        @inbounds for i in 1:m
            e, r = _ess_rhat(Float64.(view(draws, :, i, :)))
            es[i] = e; rh[i] = r
        end
        # a non-finite R-hat (e.g. a coefficient stuck at one value) counts as
        # failed mixing rather than being dropped from the maximum
        return (; draws, sum_b, sumsq_b, npos, npip, φ_chain, eps_chain, ndiv,
                acc_chain, seconds, rh, es, tag,
                rhat_max = (isempty(rh) || any(!isfinite, rh)) ? Inf : maximum(rh),
                div_frac = sum(ndiv) / (n_samples * nchains))
    end

    # Each chain already runs on its own Julia thread. Letting BLAS also spawn
    # its own threads inside every gradient oversubscribes the cores; pin BLAS
    # to one thread while chains run in parallel (restored afterwards).
    blas_before = LinearAlgebra.BLAS.get_num_threads()
    blas_used = blas_threads === :auto ?
        (nchains > 1 && Threads.nthreads() > 1 ? 1 : blas_before) : Int(blas_threads)
    LinearAlgebra.BLAS.set_num_threads(blas_used)

    local pass, retried
    try
        primary_metric = pf_metric === nothing ?
            AdvancedHMC.DiagEuclideanMetric(llcb_dimension(mdl)) : pf_metric
        pass = sampling_pass(primary_metric, pf_metric === nothing,
                             pf_metric === nothing ? :diagonal : :pathfinder)
        verbose && @info "$(now()) NUTS ($(pass.tag)): $(round(pass.seconds; digits = 1))s, rhat max $(round(pass.rhat_max; digits = 3)), divergences $(round(100 * pass.div_frac; digits = 2))%"

        retried = false
        if retry_on_poor_mixing && pass.tag === :pathfinder &&
           (pass.rhat_max > rhat_tolerance || pass.div_frac > divergence_tolerance)
            retried = true
            verbose && @info "$(now()) mixing below tolerance; retrying with an adapted diagonal metric"
            diag_metric = pf_var === nothing ?
                AdvancedHMC.DiagEuclideanMetric(llcb_dimension(mdl)) :
                AdvancedHMC.DiagEuclideanMetric(copy(pf_var))
            alt = sampling_pass(diag_metric, true, :diagonal)
            verbose && @info "$(now()) NUTS (diagonal): $(round(alt.seconds; digits = 1))s, rhat max $(round(alt.rhat_max; digits = 3)), divergences $(round(100 * alt.div_frac; digits = 2))%"
            # keep whichever pass mixed better
            score(x) = (max(x.rhat_max, 1.0) - 1.0) + 10 * x.div_frac
            score(alt) < score(pass) && (pass = alt)
        end
    finally
        # BLAS's thread count is process-wide: restore it even if sampling throws
        LinearAlgebra.BLAS.set_num_threads(blas_before)
    end

    # --- summaries ---------------------------------------------------------
    ntot = n_samples * nchains
    b_mean = vec(sum(pass.sum_b; dims = 2)) ./ ntot
    b_var = max.(vec(sum(pass.sumsq_b; dims = 2)) ./ ntot .- b_mean .^ 2, 0.0)
    b_pos = vec(sum(pass.npos; dims = 2)) ./ ntot
    b_pip = vec(sum(pass.npip; dims = 2)) ./ ntot

    B_mean = to_matrix(b_mean, p)
    quantiles = Dict{Float64,Matrix{Float64}}()
    for pr in quantile_probs
        qv = Vector{Float64}(undef, m)
        @inbounds for i in 1:m
            qv[i] = quantile(vec(Float64.(view(pass.draws, :, i, :))), pr)
        end
        quantiles[pr] = to_matrix(qv, p)
    end

    Bmap = coefficients(mdl, θmap)
    diagnostics = Dict{Symbol,Any}(
        :map_logp => llcb_logdensity(mdl, θmap),
        :map_seconds => t_map,
        :map_start => map_start,
        :blas_threads_during_sampling => blas_used,
        :pathfinder_seconds => t_pf,
        :sample_seconds => pass.seconds,
        :metric => pass.tag,
        :retried => retried,
        :step_size => pass.eps_chain,
        :divergences => pass.ndiv,
        :divergence_fraction => pass.div_frac,
        :acceptance => pass.acc_chain,
        :rhat_max => pass.rhat_max,
        :phi => mean(pass.φ_chain),
        :spectral_radius_posterior_mean => maximum(abs.(eigvals(B_mean))),
        :spectral_radius_map => maximum(abs.(eigvals(Bmap))),
        :nchains => nchains, :n_samples => n_samples, :n_adapt => n_adapt,
        :thin => thin, :optim => optres, :pathfinder => pf_result,
    )

    fit = LLCBFit(
        B_mean, to_matrix(sqrt.(b_var), p), to_matrix(b_pos, p),
        to_matrix(min.(b_pos, 1 .- b_pos), p), to_matrix(b_pip, p), quantiles,
        to_matrix(pass.rh, p), to_matrix(pass.es, p), Bmap, pass.draws,
        diagnostics, te.targets, te.perturbed, pip_threshold,
    )
    return fit, mdl
end

"""
    fit_llcb_closed_form(te::TotalEffectEstimate; ndraws = 4_000, rng, kwargs...) -> LLCBFit

Direct effects without MCMC: draw total-effect matrices from the stage-1
sampling distribution and push each through [`closed_form_coefficients`](@ref).

Each draw costs one `p × p` inversion, so this scales to thousands of genes.
It needs every readout to have been perturbed. Two limitations relative to
[`fit_llcb`](@ref) are worth keeping in mind: there is no shrinkage prior, so
every edge gets a non-zero estimate and `lsfr` is the only sparsity signal; and
the intervals reflect stage-1 sampling error only, with no model-error
dispersion `φ`.

Rows are resampled with the full within-knockout covariance when `te` carries
it (`estimate_total_effects(...; covariance = :full)`) and with independent
per-entry standard errors otherwise. Draws whose total-effect matrix is
singular are discarded and counted in `diagnostics[:singular_draws]`.
"""
function fit_llcb_closed_form(
        te::TotalEffectEstimate;
        ndraws::Int = 4_000,
        rng::Random.AbstractRNG = Random.default_rng(),
        pip_threshold::Float64 = 0.05,
        quantile_probs::Vector{Float64} = [0.025, 0.5, 0.975],
        max_stored::Int = 4_000,
        verbose::Bool = true,
    )
    has_full_total_effects(te) || throw(ArgumentError(
        "fit_llcb_closed_form needs every readout perturbed with an identifiable knockdown; use fit_llcb"))
    p = size(te.Θ, 1); m = p * (p - 1)
    t0 = time()
    B̂ = closed_form_coefficients(te)

    # per-row sampling factors
    use_full = te.Σ !== nothing
    L = Vector{Matrix{Float64}}(undef, p)
    if use_full
        for k in 1:p
            S = Symmetric(te.Σ[k])
            ev = eigen(S)
            L[k] = ev.vectors * Diagonal(sqrt.(max.(ev.values, 0.0)))
        end
    end

    nstored = min(ndraws, max_stored)
    draws = Array{Float32,3}(undef, nstored, m, 1)
    sum_b = zeros(m); sumsq_b = zeros(m); npos = zeros(Int, m); npip = zeros(Int, m)
    nsing = 0; kept = 0; store = 0
    Θs = similar(te.Θ)
    I_p = Matrix{Float64}(I, p, p)
    while kept < ndraws
        if use_full
            for k in 1:p
                Θs[k, :] .= view(te.Θ, k, :) .+ L[k] * randn(rng, p)
            end
        else
            Θs .= te.Θ .+ te.se .* randn(rng, p, p)
        end
        for u in 1:p; Θs[u, u] = 0.0; end
        local Bd
        try
            Bd = closed_form_coefficients(Θs + I_p)
        catch
            nsing += 1
            nsing > 10 * ndraws && error("total-effect draws are singular too often; check stage 1")
            continue
        end
        kept += 1
        b = from_matrix(Bd)
        @inbounds for i in 1:m
            sum_b[i] += b[i]; sumsq_b[i] += b[i]^2
            b[i] > 0 && (npos[i] += 1)
            abs(b[i]) > pip_threshold && (npip[i] += 1)
        end
        if store < nstored
            store += 1
            @views draws[store, :, 1] .= Float32.(b)
        end
    end

    b_mean = sum_b ./ ndraws
    b_var = max.(sumsq_b ./ ndraws .- b_mean .^ 2, 0.0)
    b_pos = npos ./ ndraws
    quantiles = Dict{Float64,Matrix{Float64}}()
    for pr in quantile_probs
        quantiles[pr] = to_matrix([quantile(vec(Float64.(view(draws, :, i, 1))), pr) for i in 1:m], p)
    end
    secs = time() - t0
    verbose && @info "$(now()) closed-form resampling: $ndraws draws in $(round(secs; digits = 2))s ($nsing singular draws discarded)"

    diagnostics = Dict{Symbol,Any}(
        :method => :closed_form_resampling,
        :covariance => use_full ? :full : :diagonal,
        :ndraws => ndraws, :singular_draws => nsing, :seconds => secs,
        :spectral_radius_point => maximum(abs.(eigvals(B̂))),
    )
    return LLCBFit(
        to_matrix(b_mean, p), to_matrix(sqrt.(b_var), p), to_matrix(b_pos, p),
        to_matrix(min.(b_pos, 1 .- b_pos), p), to_matrix(npip ./ ndraws, p), quantiles,
        fill(NaN, p, p), fill(Float64(ndraws), p, p), B̂, draws,
        diagnostics, te.targets, te.perturbed, pip_threshold,
    )
end

"""
    parse_llcb_fit(fit::LLCBFit; targets) -> DataFrame

Long-format edge table: one row per ordered pair, with the posterior mean,
credible interval, `PIP`, `lsfr`, `rhat` and `ess`. Column names match the
output of the original `parse_cyclic_chain` so downstream code keeps working.
"""
function parse_llcb_fit(fit::LLCBFit; targets::Vector{String} = fit.targets)
    p = size(fit.B_mean, 1)
    length(targets) == p ||
        throw(ArgumentError("targets has length $(length(targets)) but B is $p × $p"))
    lo = get(fit.quantiles, 0.025, fill(NaN, p, p))
    hi = get(fit.quantiles, 0.975, fill(NaN, p, p))
    result = DataFrame()
    for j in 1:p, i in 1:p
        i == j && continue
        push!(result, (;
            :row => targets[i], :col => targets[j],
            :estimate => fit.B_mean[i, j],
            Symbol("2.5%") => lo[i, j], Symbol("97.5%") => hi[i, j],
            :PIP => fit.pip[i, j], :lsfr => fit.lsfr[i, j],
            :prob_pos => fit.prob_pos[i, j],
            :rhat => fit.rhat[i, j], :ess => fit.ess[i, j],
            :std => fit.B_sd[i, j], :map => fit.map_B[i, j],
        ))
    end
    return result
end
