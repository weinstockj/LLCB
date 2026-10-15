
# =============================================================================
# Top-level entry point for LLCB.
# =============================================================================

"""
    fit_cyclic_model(g::interventionGraph; kwargs...)

Fit the LLCB cyclic model and return `(fit, model, total_effects)`.

Two stages:

1. [`estimate_total_effects`](@ref) forms donor-adjusted interventional total
   effects together with their sampling uncertainty.
2. [`fit_llcb`](@ref) fits the direct-effect matrix `B` against the exact
   moment condition `T̃ = Diagonal(diag(A))⁻¹ A`, `A = (I - B)⁻¹`, weighted by
   that uncertainty, by NUTS on an analytic gradient.

```julia-repl
julia> graph = interventionGraph(expression);

julia> fit, model, te = fit_cyclic_model(graph);

julia> edges = parse_llcb_fit(fit);
```

# Keyword arguments
Stage 1: `log_normalize`, `center`, `scale`, `covariance`, `min_knockdown`.
Stage 2: `prior`, `prior_scale`, `prior_df`, `rho_max`, `nchains`, `n_adapt`,
`n_samples`, `target_accept`, `use_pathfinder`, `pip_threshold`, `seed`.
All stage-2 arguments are forwarded to [`fit_llcb`](@ref).
"""
function fit_cyclic_model(
        g::interventionGraph;
        log_normalize::Bool = false,
        center::Bool = true,
        scale::Bool = true,
        covariance::Symbol = :diagonal,
        min_knockdown::Float64 = 0.25,
        verbose::Bool = true,
        kwargs...,
    )
    te = estimate_total_effects(
        g;
        log_normalize = log_normalize, center = center, scale = scale,
        covariance = covariance, min_knockdown = min_knockdown, verbose = verbose,
    )
    fit, model = fit_llcb(te; verbose = verbose, kwargs...)
    return (fit = fit, model = model, total_effects = te)
end

"""
    fit_cyclic_model(g, log_normalize, model_pars, sampling_pars)

Compatibility method for the published call signature. The old `model_pars`
and `sampling_pars` named tuples are mapped onto their closest equivalents:
`:l1_penalty` sets the prior scale, `:warmups` the adaptation length and
`:samples` the number of retained draws. `:R_scale`, `:l2_penalty`,
`:β_sum_scale`, `:θ_scale`, `:ω_scale` and `:σₓ_scale` no longer have
counterparts -- donor and gene offsets are handled in stage 1 and the noise
scale is estimated rather than assumed -- and are ignored.
"""
function fit_cyclic_model(
        g::interventionGraph, log_normalize::Bool,
        model_pars::NamedTuple, sampling_pars::NamedTuple; kwargs...,
    )
    @warn "the positional fit_cyclic_model signature is deprecated; see the keyword method"
    return fit_cyclic_model(
        g;
        log_normalize = log_normalize,
        prior_scale = get(model_pars, :l1_penalty, 0.1),
        n_adapt = get(sampling_pars, :warmups, 500),
        n_samples = get(sampling_pars, :samples, 500),
        target_accept = get(sampling_pars, :acceptance, 0.8),
        max_depth = get(sampling_pars, :max_depth, 10),
        kwargs...,
    )
end
