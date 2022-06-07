@model function joint_cyclic_model(nv, pars, T, t, ::Type{S} = Float64) where {S}
    
    β ~ filldist(Laplace(0, 3.0), nv * (nv - 1))
    # β ~ filldist(Normal(0, 1.0), nv * (nv - 1))
    W = Array{S}(undef, nv, nv) # I - the entire adjacency matrix
    σₓ ~ Normal(-2.0, 1.0)

    for i in 1:nv
        for j in 1:nv
            if i == j
                W[i, j] = 0
            else
                @assert i - Int(i > j) <= (nv - 1)
                # W[i, j] = β[i - Int(i > j) + j]
                alt_idx = (i - 1) * (nv - 1) + j - Int(j > i)
                W[i, j] = β[alt_idx]
            end
        end
    end
    Wdet = det(W)
    
    Turing.@addlogprob! loglikelihood(Normal(0.0, 0.3), Wdet)
    t ~ MvNormal(T*β, I * exp(σₓ))
    # t .~ Normal.(T*β, 0.1) 
    return W
end

@model function joint_model_discrete_mcmc(x, n, nv, donors, n_donors, interventions, μ, pars, ::Type{T} = Float64) where {T}

    β ~ DAG(nv)

    θ ~ filldist(Normal(0, pars[:θ_scale]), nv)
    ω ~ filldist(Normal(0, pars[:ω_scale]), n_donors)

    # println("β = $β")
    scorer = DAGScorer(BitMatrix(β))
    bias = Array{T}(undef, n, nv)
    for i in 1:n
        for j in 1:nv
            bias[i, j] = μ + θ[j] + ω[donors[i]]
        end
    end

    scores = Array{T}(undef, nv)
    for j in 1:nv
        observed = interventions[:, j] .!= 1
        perturbed_score = bge(scorer, x[observed, :] .- bias[observed, :]) / nv
        # println("perturbed_score = $perturbed_score")
        Turing.@addlogprob! perturbed_score
        scores[j] = perturbed_score
    end

    return β
end

@model function joint_model_mcmc(x, n, nv, donors, n_donors, interventions, μ, pars, ::Type{T} = Float64) where {T}
    β ~ filldist(Laplace(0, pars[:β_scale]), nv, nv) # this skips the diagonal elements
    l1 = norm(β, 1)
    l2 = norm(β, 2)
    Turing.@addlogprob! loglikelihood(Normal(0.0, pars[:l1_penalty]), l1 * nv * (nv - 1) / 20)
    Turing.@addlogprob! loglikelihood(Normal(0.0, pars[:l2_penalty]), l2 * nv * (nv - 1) / 20) 

    θ ~ filldist(Normal(0, pars[:θ_scale]), nv)
    ω ~ filldist(Normal(0, pars[:ω_scale]), n_donors)
    # σₓ ~ truncated(Cauchy(0, pars[:σₓ_scale]), 0, Inf)
    σₓ ~ Normal(-2.0, 1.0)


    λₓ = Array{T}(undef, n, nv)
    for i in 1:n
        for j in 1:nv
            vec = 1:nv .!= j
            indices = (1:nv)[vec]
            ν = sum(β[indices, j] .* x[i, indices])

            λₓ[i, j] = μ + θ[j] + ω[donors[i]] + ν
            if interventions[i, j] != 1 # only count instances when target gene is not perturbed
                x[i, j] ~ Normal(λₓ[i, j], exp(σₓ) + .0001)
            end
        end
    end

    return β
end
   


@model function joint_model(x, n, nv, donors, n_donors, interventions, μ, pars, ::Type{T} = Float64) where {T}
   
    # β ~ filldist(Laplace(0, 0.1), nv - 1, nv)
    # β = Array{T}(undef, nv, nv)

    # global_scale_a ~ truncated(Normal(0, 0.1), 0, Inf)
    # global_scale_b ~ InverseGamma(0.5, 0.5)
    # τ = global_scale_a * sqrt(global_scale_b)
    # τ ~ truncated(TDist(3), 0, Inf)

    # local_scale_a ~ filldist(truncated(Normal(0, 0.1), 0, Inf), nv - 1, nv)
    # local_scale_b ~ filldist(InverseGamma(0.5, 0.5), nv - 1, nv)
    #
    # λ = local_scale_a .* sqrt.(local_scale_b)
    # λ = Array{T}(undef, nv, nv - 1)

    # λ .~ truncated(TDist(3), 0, Inf)
    # β ~ Normal.(0, .01 * τ * λ)
    # for i in 1:(nv - 1)
    #     for j in 1:nv
    #         β[i, j] ~ Normal(0, .001 + τ * λ[i, j])
    #     end
    # end
    #
    # β .~ Normal(0, .01)

    # β .~ Normal.(0, .01 .* λ .* τ)
    # 
    # for i in 1:nv
    #     for j in 1:nv
    #         # β[i, j] ~ Normal(0, 0.005)
    #         if i == j
    #             β[i, j] ~ Normal(0.0, .0001)
    #         else
    #             β[i, j] ~ MixtureModel(Normal, [(0.0, 0.001), (0.0, 0.10)], [0.90, 0.10])
    #         end
    #     end
    # end
    β ~ filldist(Laplace(0, pars[:β_scale]), nv - 1, nv) # this skips the diagonal elements
    # β ~ filldist(MixtureModel(Normal, [(0.0, 0.0001), (0.0, 0.10)], [0.98, 0.02]), nv, nv)
    # β ~ filldist(UnivariateGMM([0.0, 0.0], [0.02, 0.30], Categorical([0.6, 0.4])), nv - 1, nv)
    W = Array{T}(undef, nv, nv) # I - the entire adjacency matrix

    for i in 1:nv
        for j in 1:nv
            if i == j
                W[i, j] = 0
            else
                @assert i - Int(i > j) <= (nv - 1)
                W[i, j] = β[i - Int(i > j), j]
            end
        end
    end


    # R = det(W) # I - adjacency
    # R = notears(W)
    R = spectral_radius(W)
    # R ~ Normal(0, 0.01)
    # if rand(Bernoulli(.004), 1)[1]
        # println("W = $W")
        # println("R = $R")
    # end
    Turing.@addlogprob! loglikelihood(Normal(0.0, pars[:R_scale]), R)
    # OP = opnorm(W, 1)
    l1 = norm(W, 1)
    l2 = norm(W, 2)
    Turing.@addlogprob! loglikelihood(Normal(0.0, pars[:l1_penalty]), l1 * nv * (nv - 1) / 20)
    Turing.@addlogprob! loglikelihood(Normal(0.0, pars[:l2_penalty]), l2 * nv * (nv - 1) / 20) 
    # β_sum = sum(β)
    # Turing.@addlogprob! loglikelihood(Normal(0.0, pars[:β_sum_scale]), β_sum)
    #
    # σ_θ ~ truncated(Normal(0, 1.0), 0, Inf)
    #
    θ ~ filldist(Normal(0, pars[:θ_scale]), nv)
    # θ ~ MvNormal(zeros(nv), I * .001)
    # θ ~ filldist(Normal(0, σ_θ), nv)

    
    # σ_w ~ truncated(Normal(0, 1), 0, Inf)
    ω ~ filldist(Normal(0, pars[:ω_scale]), n_donors)
    # ω ~ MvNormal(zeros(n_donors), I * .001)

    # σₓ ~ truncated(Cauchy(0, pars[:σₓ_scale]), 0, Inf)
    σₓ ~ Normal(-2, 1.0) # log scale

    λₓ = Array{T}(undef, n, nv)
    λₓ
    for i in 1:n
        for j in 1:nv
            vec = 1:nv .!= j
            indices = (1:nv)[vec]
            ν = sum(β[:, j] .* x[i, indices])
            # ν = sum(β[indices, j] .* x[i, indices])

            # gene_vec = ones(nv) .* (1:nv .== j)

            λₓ[i, j] = μ + θ[j] + ω[donors[i]] + ν
            # λₓ[i, j] = μ + θ'gene_vec + ω[donors[i]] + ν
            # if rand(Bernoulli(.000005), 1)[1]
            #     println("j = $(j); indices = $(indices)")
            #     println("μ = $(μ); θ[j] = $(θ[j]); ν = $(ν)")
            #     println("intervene[$i, $j] = $(interventions[i, j]); x[$i, $j] =$(x[i, j]); λ[$i, $j] = $(exp(λ[i, j]))")
            # end
            # Turing.@addlogprob! loglikelihood(Normal(μ + θ[j] + ω[donors[i]] + sum(β[indices, j] .* x[i, indices]), σₓ + .0001), x[i, j]) * (1.0 - interventions[i, j])
            if interventions[i, j] != 1 # only count instances when target gene is not perturbed
                # x[i, j] ~ LogNormal(λₓ[i, j], σₓ + .0001)
                x[i, j] ~ Normal(λₓ[i, j], exp(σₓ) + .0001)
                # x[i, j] ~ Normal(μ + θ'gene_vec + ω'donors[i, :] + sum(β[indices, j] .* x[i, indices]), σₓ + .0001)
                # x[i, j] ~ Normal(μ + θ'gene_vec + ω'donors[i, :] + sum(β[:, j] .* (x[i, :]'vec)), σₓ + .0001)

            end
        end
    end

    return W

end
