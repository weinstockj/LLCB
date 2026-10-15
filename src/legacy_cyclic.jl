# =============================================================================
# The published LLCB stage 2 (Weinstock & Arce et al.), ported to Turing 0.46 /
# DynamicPPL 0.42 / Pathfinder 0.10.
#
# The settings match the published analysis:
#
#   1. Each Pathfinder run is initialised with `Pathfinder.UniformSampler(0.03)`,
#      i.e. Uniform(-0.03, 0.03) in every coordinate, as in
#      `wrap_multipathfinder`.
#   2. The optimizer is `Optim.LBFGS(; m = 8)`, as in `wrap_multipathfinder`.
#
# `init = :published` instead starts from the moment-based point computed in
# `fit_cyclic_model` (`0.3 .* (T \ t)` for β and -3 for each log σ, plus
# N(0, 0.1) jitter), for anyone who wants to test whether the start matters.
#
# What changed is mechanics only: `Turing.setadbackend` and
# `Turing.optim_function` no longer exist, and Pathfinder now accepts a Turing
# model directly and returns `draws_transformed` as an `MCMCChains.Chains`, so
# the hand-written wrapper is no longer needed. `generated_quantities` is now
# `DynamicPPL.returned`.
#
# For exact reproduction of the published numbers use the v0.2.0b tag on
# Julia 1.9.4; this port changes the RNG stream and the dependency versions.
# =============================================================================

"""
    fit_cyclic_model_legacy(g, log_normalize, model_pars, sampling_pars; kwargs...)

The published LLCB stage 2: build the block system with
[`get_cyclic_matrices`](@ref), fit [`joint_cyclic_model`](@ref) by multi-path
Pathfinder, and report Pathfinder's approximation as the posterior.

`model_pars` and `sampling_pars` are accepted for signature compatibility; as
in the published analysis, the priors are those set in `joint_cyclic_model`
and the posterior is Pathfinder's approximation.

# Keyword arguments
- `rng = Random.default_rng()`: the published code used the global RNG.
- `init = :uniform`: `:uniform` matches the published analysis (every run
  starts from Uniform(-0.03, 0.03)); `:published` starts from the moment-based
  point computed in `fit_cyclic_model`.
- `nruns = 5`, `ndraws = 2_000`, `ndraws_elbo = 100`, `ntries = 1_500`,
  `importance = true`: as published.
- `adtype = AutoForwardDiff()`: the published code set the ForwardDiff backend.

Returns `(chains, posterior_adjacency, model, pathfinder_result)` as before, so
`parse_cyclic_chain(result[1], result[2], edges)` keeps working.
"""
function fit_cyclic_model_legacy(
        g::interventionGraph, log_normalize::Bool,
        model_pars::NamedTuple, sampling_pars::NamedTuple;
        rng::Random.AbstractRNG = Random.default_rng(),
        init::Symbol = :uniform,
        nruns::Int = 5,
        ndraws::Int = 2_000,
        ndraws_elbo::Int = 100,
        ntries::Int = 1_500,
        importance::Bool = true,
        adtype = ADTypes.AutoForwardDiff(),
    )
    init in (:uniform, :published) ||
        throw(ArgumentError("init must be :uniform or :published, got $init"))

    @info "$(now()) preparing matrices for the model"
    T, t, edges = get_cyclic_matrices(g, log_normalize, true, true)
    T = sparse(T)
    model = joint_cyclic_model(g.nv, model_pars, T, t)
    dim = g.nv * (g.nv - 1) + g.nv          # β, then one log σ per child block

    inits = if init === :uniform
        [rand(rng, dim) .* 0.06 .- 0.03 for _ in 1:nruns]
    else
        base = vcat(0.3 .* (T \ t), fill(-3.0, g.nv))
        [base .+ 0.10 .* randn(rng, dim) for _ in 1:nruns]
    end

    @info "$(now()) running pathfinder now"
    result = Pathfinder.multipathfinder(
        model, ndraws;
        nruns = nruns, init = inits, rng = rng, adtype = adtype,
        optimizer = Optim.LBFGS(; m = 8),
        ndraws_elbo = ndraws_elbo, importance = importance, ntries = ntries,
        chain_type = MCMCChains.Chains,
    )
    @info "$(now()) done running pathfinder"

    model_chain = result.draws_transformed
    chains_params = MCMCChains.get_sections(model_chain, :parameters)
    quantities = DynamicPPL.returned(model, chains_params)
    return model_chain, quantities, model, result
end

"""
`parse_cyclic_chain(chains, posterior_adjacency, edges)`

```julia-repl
    model = fit_cyclic_model_legacy(graph, false, model_pars, sampling_pars)
    parsed = parse_cyclic_chain(
        model[1], model[2], cyclic_matrices[3]
    )
```

"""
function parse_cyclic_chain(chains::MCMCChains.Chains, posterior_adjacency::AbstractMatrix, edges::AbstractVector{<:Pair}; targets=setdiff(ko_targets(), ko_controls()))

    n_samples = size(posterior_adjacency, 1) # slightly misleading - total mcmc samples is n_samples * n_chains
    n_chains  = size(posterior_adjacency, 2)

    @assert n_samples >= 10
    @assert n_chains  >= 1

    posterior_mean = mean(mean(posterior_adjacency; dims = 2)) # inner mean takes mean across chains, outer averages across samples
    ϵ = .05
    nv = size(posterior_mean, 1)
    @assert nv == size(posterior_mean, 2)

    posterior_pip = mean(
        abs.(
            reduce(hcat, vec.(vec(posterior_adjacency)))
        ) .> ϵ;
        dims = 2
    )
    posterior_pip = reshape(posterior_pip, nv, nv)

    posterior_lsfr_pos = mean(
        (
            reduce(hcat, vec.(vec(posterior_adjacency)))
        ) .>= 0;
        dims = 2
    )
    posterior_lsfr_neg = mean(
        (
            reduce(hcat, vec.(vec(posterior_adjacency)))
        ) .< 0;
        dims = 2
    )

    posterior_lsfr = min.(posterior_lsfr_pos, posterior_lsfr_neg)
    posterior_lsfr = reshape(posterior_lsfr, nv, nv)

    result = DataFrame() # convert to 'long' format

    for (z, e) in enumerate(edges)
        i = e.first
        j = e.second
        row_label = targets[i]
        col_label = targets[j]

            if i == j

                row = (;
                    :row => row_label,
                    :col => col_label,
                    :estimate => 0.0,
                    Symbol("2.5%") => 0.0,
                    Symbol("97.5%") => 0.0,
                    :PIP => 0.0,
                    :lsfr => 1.0,
                    :rhat => 1.0,
                    :ess => Float64(n_samples * n_chains),
                    :std => 0.0
                )

            else
                # (1, 2) -> (1)
                # 2 - Int(2 > 1) == 1
                # (3, 1) -> (2) * (nv - 1) + 1 - Int(1 > 3) = 2 * nv - 2 + 1 - 0
                # alt_idx = (i - 1) * (nv - 1) + j - Int(j > i)
                alt_idx = z
                summary   = summarize(chains[[Symbol("β[$alt_idx]")]])
                quantiles = quantile(chains[[Symbol("β[$alt_idx]")]])

                row = (;
                    :row => row_label,
                    :col => col_label,
                    :estimate => summary[:, :mean],
                    Symbol("2.5%") => quantiles[:, Symbol("2.5%")],
                    Symbol("97.5%") => quantiles[:, Symbol("97.5%")],
                    :PIP => posterior_pip[j, i],
                    :lsfr => posterior_lsfr[j, i],
                    :rhat => summary[:, :rhat],
                    :ess => summary[:, :ess_bulk],
                    :std => summary[:, :std]
                )

            end

            push!(result, row)
        # end
    end

    return result
end
