# =============================================================================
# Turing models from the published LLCB implementation, ported to
# Turing 0.46 / DynamicPPL 0.42.
#
# Changes are confined to API mechanics; the statistical model is unchanged.
#   * `::Type{T} = Float64` model arguments and `Array{T}(undef, ...)` buffers
#     are replaced by buffers whose element type follows the sampled
#     parameters, which is how current DynamicPPL propagates AD dual numbers.
#   * `MvNormal(μ, σ::AbstractVector)` was removed from Distributions. In the
#     pinned Distributions 0.25.104 it was deprecated to
#     `MvNormal(μ, Diagonal(abs2.(σ)))`, i.e. the vector was read as standard
#     deviations; that is written out explicitly here so the noise model is
#     preserved exactly.
#   * `Turing.@addlogprob!` is now exported as `@addlogprob!`.
#
# `joint_model_discrete_mcmc` (the discrete-DAG model) is not ported: it
# depends on `DAG`, `DAGScorer` and `bge` from DAGsampler.jl / DAGproposal.jl,
# which were not part of the v0.2.0b module build or the published analysis.
# =============================================================================

"""
    joint_cyclic_model(nv, pars, T, t)

The published LLCB stage-2 model: `t ~ MvNormal(T β, Diagonal(exp.(σ)²))` over
the block system built by [`get_cyclic_matrices`](@ref), with a `Normal(0, 1)`
prior on each direct effect, a spectral-radius penalty and a column-sum
penalty on the adjacency matrix. Returns the `nv × nv` adjacency matrix `W`.

`W[a, b] = β[(a - 1)(nv - 1) + b - (b > a)]`, so with `edges` from
`get_cyclic_matrices`, coefficient `β[z]` for edge `edges[z] = (i => j)` is
`W[j, i]`; [`parse_cyclic_chain`](@ref) reads it that way.

Note that `spectral_radius` seeds power iteration with `Base.rand`, so this
model's log density is not a deterministic function of its parameters. That is
kept as published.
"""
@model function joint_cyclic_model(nv::Int, pars, T, t)
    β ~ filldist(Normal(0, 1.0), nv * (nv - 1))
    σₓ ~ filldist(Normal(-3, 5), nv)               # log standard deviations
    temp_σₓ = reduce(vcat, (fill(x, nv - 1) for x in σₓ))

    W = zeros(eltype(β), nv, nv)
    @inbounds for i in 1:nv, j in 1:nv
        i == j && continue
        W[i, j] = β[(i - 1) * (nv - 1) + j - Int(j > i)]
    end

    R = spectral_radius(W)
    @addlogprob! loglikelihood(Normal(0.0, 10.0), R)
    # penalise the number of parents assigned to each child node
    @addlogprob! sum(loglikelihood(Normal(0.0, nv * 0.1), x) for x in vec(sum(abs.(W); dims = 1)))

    t ~ MvNormal(T * β, Diagonal(exp.(temp_σₓ) .^ 2))
    return W
end

"""
    joint_model_mcmc(x, n, nv, donors, n_donors, interventions, μ, pars)

Published direct-likelihood model on expression data (Laplace prior on `β`,
L1/L2 penalties, gene and donor offsets). Not used by the LLCB cyclic fit.
"""
@model function joint_model_mcmc(x, n, nv, donors, n_donors, interventions, μ, pars)
    β ~ filldist(Laplace(0, pars[:β_scale]), nv, nv)
    l1 = norm(β, 1)
    l2 = norm(β, 2)
    @addlogprob! loglikelihood(Normal(0.0, pars[:l1_penalty]), l1 * nv * (nv - 1) / 20)
    @addlogprob! loglikelihood(Normal(0.0, pars[:l2_penalty]), l2 * nv * (nv - 1) / 20)

    θ ~ filldist(Normal(0, pars[:θ_scale]), nv)
    ω ~ filldist(Normal(0, pars[:ω_scale]), n_donors)
    σₓ ~ Normal(-2.0, 1.0)

    S = promote_type(eltype(β), eltype(θ), eltype(ω), typeof(σₓ))
    λₓ = zeros(S, n, nv)
    for i in 1:n, j in 1:nv
        indices = (1:nv)[1:nv .!= j]
        ν = sum(β[indices, j] .* x[i, indices])
        λₓ[i, j] = μ + θ[j] + ω[donors[i]] + ν
        if interventions[i, j] != 1      # only count samples where gene j is not perturbed
            x[i, j] ~ Normal(λₓ[i, j], exp(σₓ) + .0001)
        end
    end
    return β
end

"""
    joint_model(x, n, nv, donors, n_donors, interventions, μ, pars)

Published direct-likelihood model with the spectral-radius penalty on the
reduced `(nv - 1) × nv` coefficient layout. Not used by the LLCB cyclic fit.
"""
@model function joint_model(x, n, nv, donors, n_donors, interventions, μ, pars)
    β ~ filldist(Laplace(0, pars[:β_scale]), nv - 1, nv)   # skips the diagonal
    W = zeros(eltype(β), nv, nv)
    for i in 1:nv, j in 1:nv
        i == j && continue
        W[i, j] = β[i - Int(i > j), j]
    end

    R = spectral_radius(W)
    @addlogprob! loglikelihood(Normal(0.0, pars[:R_scale]), R)
    l1 = norm(W, 1)
    l2 = norm(W, 2)
    @addlogprob! loglikelihood(Normal(0.0, pars[:l1_penalty]), l1 * nv * (nv - 1) / 20)
    @addlogprob! loglikelihood(Normal(0.0, pars[:l2_penalty]), l2 * nv * (nv - 1) / 20)

    θ ~ filldist(Normal(0, pars[:θ_scale]), nv)
    ω ~ filldist(Normal(0, pars[:ω_scale]), n_donors)
    σₓ ~ Normal(-2, 1.0)       # log scale

    S = promote_type(eltype(β), eltype(θ), eltype(ω), typeof(σₓ))
    λₓ = zeros(S, n, nv)
    for i in 1:n, j in 1:nv
        indices = (1:nv)[1:nv .!= j]
        ν = sum(β[:, j] .* x[i, indices])
        λₓ[i, j] = μ + θ[j] + ω[donors[i]] + ν
        if interventions[i, j] != 1
            x[i, j] ~ Normal(λₓ[i, j], exp(σₓ) + .0001)
        end
    end
    return W
end
