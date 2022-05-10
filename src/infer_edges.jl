Turing.setprogress!(true)

function get_model_params(simulation::Bool, l1_penalty = 0.01, R_scale = 0.05)

    # TODO: maybe use dictionary here instead of named tuple
   vals = (;
        :β_scale => simulation ? 0.00070 : 0.00050,
        :R_scale => simulation ? 0.005 : R_scale,
        :l1_penalty => simulation ? 0.02 : l1_penalty, # smaller numbers mean more penalty
        :l2_penalty => simulation ? 0.01 : 0.01, # smaller numbers mean more penalty
        :β_sum_scale => simulation ? 0.008 : 0.01,
        :θ_scale => simulation ? 0.01 : 0.10,
        :ω_scale => simulation ? 0.01 : 0.05,
        :σₓ_scale => simulation ? 0.40 : 0.50
   )

   return vals
end

function get_sampling_params(simulation::Bool)

   vals = (;
        :warmups => simulation ? 400 : 500,
        :acceptance => simulation ? 0.85 : 0.90,
        :max_depth => simulation ? 4 : 5,
        :init_ϵ => simulation ? 0.01 : 0.01,
        :samples => simulation ? 200 : 300
   )

   return vals
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

scale(A) = (A .- mean(A,dims=1)) ./ std(A,dims=1)

function normalize(x::Matrix)
    return log1p.(x)
end

function one_hot(d::Vector{Int8})
    return transpose(unique(d) .== permutedims(d))
end

function fit_model(g::interventionGraph, log_normalize::Bool, model_pars::NamedTuple, sampling_pars::NamedTuple, discrete = true)
    Turing.setadbackend(:forwarddiff)

    if log_normalize
        μ = mean(normalize(g.x) .* (1.0 .- g.interventions))
        # model = joint_model(normalize(g.x), size(g.x, 1), g.nv, one_hot(g.donor), length(unique(g.donor)), g.interventions, μ, 0.0)
        model = joint_model(normalize(g.x), size(g.x, 1), g.nv, g.donor, length(unique(g.donor)), g.interventions, μ, model_pars)
        # model_mcmc = joint_model_mcmc(normalize(g.x), size(g.x, 1), g.nv, g.donor, length(unique(g.donor)), g.interventions, μ, model_pars)
        model_mcmc = joint_model_discrete_mcmc((normalize(g.x)), size(g.x, 1), g.nv, g.donor, length(unique(g.donor)), g.interventions, μ, model_pars)
    else
        μ = mean(g.x .* (1.0 .- g.interventions))
        model = joint_model(g.x, size(g.x, 1), g.nv, g.donor, length(unique(g.donor)), g.interventions, μ, model_pars)
    end

    # @code_warntype model.f(
    #         model,
    #         Turing.VarInfo(model),
    #         Turing.SamplingContext(
    #             Random.GLOBAL_RNG, Turing.SampleFromPrior(), Turing.DefaultContext(),
    #             ),
    #         model.args...,
    #     )
    
    # model_chain = sample(model, NUTS(0.65), MCMCThreads(), 200, 4)
    # model_chain = sample(model, NUTS(0.65), 1)
    init = vcat(
        zeros(g.nv * (g.nv - 1)), #beta
        zeros(g.nv), # θ
        zeros(length(unique(g.donor))), # ω,
        log(0.1) # σₓ
    )
    rng = MersenneTwister(1)
    obj = optim_objective(model, MAP(); constrained=false)
    # init = obj.init()
    logp(x) = -obj.obj(x)
    gradlogp(x) = ForwardDiff.gradient(logp, x)
    # gradlogp(x) = Zygote.gradient(logp, x)
    # @. init = rand(rng) * 1.0 - 0.5
    @info "$(now()) running pathfinder"
    noise_dist = Normal(0, .001)
    # n_runs = 5
    # total_draws = 400
    n_runs = 2
    total_draws = 40
    q, ψs, logψs = multipathfinder(
        logp, gradlogp, 
        [init .+ rand(noise_dist, 1) for i in 1:n_runs], total_draws;
        ndraws_per_run = 20,
        importance = false,
        rng = rng,
        optimizer = Optim.LBFGS(; m = 5, linesearch = MoreThuente()),
        ndraws_elbo = 10,
        show_trace = false,
        iterations = 600
        # iterations = 300
    )
    path_init = obj.transform(ψs[:, 1])
    β_indices = 1:(g.nv * (g.nv - 1))
    θ_indices = (maximum(β_indices) + 1):(maximum(β_indices) + g.nv)
    ω_indices = (maximum(θ_indices) + 1):(maximum(θ_indices) + length(unique(g.donor)))
    σₓ_index = length(path_init)

    # takes 928 seconds wall time, 3580 seconds compute duration
    @info "$(now()) running MCMC now"
    sampler = (
        discrete ? 
        discrete_sampler(
                    g.nv,
                    length(unique(g.donor)),
                    daggify(convert_to_full_adjacency(reshape(path_init[β_indices], g.nv - 1, g.nv))),
                    path_init[θ_indices],
                    path_init[ω_indices],
                    # path_init[σₓ_index]
                    normalize(g.x),
                    g.interventions
                )
        :
        NUTS(
            sampling_pars[:warmups], sampling_pars[:acceptance];
            max_depth = sampling_pars[:max_depth], init_ϵ = sampling_pars[:init_ϵ]
        ) 
    )

    vec_β_init = vec(daggify(convert_to_full_adjacency(reshape(path_init[β_indices], g.nv - 1, g.nv))))

    path_init = vcat(
        vec_β_init .!= 0,
        path_init[θ_indices],
        path_init[ω_indices]
    )

    model_chain = (
        discrete ? 
        sample(
            model_mcmc, 
            sampler,
            # MCMCThreads(), 
            MCMCSerial(), 
            500,
            2;
            # Threads.nthreads();
            # init_params = Iterators.repeated(path_init), # doesn't work with MH sampler to specify initial state
            init_params = path_init,
            # discard_initial=2000,
            thinning = 10
        )
        :
        sample(
            model_mcmc, 
            sampler,
            MCMCThreads(), 
            sampling_pars[:samples], 
            Threads.nthreads();
            init_params = path_init
        )
    )

    # 253 seconds with forwarddiff
    chains_params = Turing.MCMCChains.get_sections(model_chain, :parameters)
    quantities = generated_quantities(model_mcmc, chains_params)
    acceptance_rate = sampler.proposals[:β].accepted / sampler.proposals[:β].total
    println("acceptance_rate = $acceptance_rate")
    return model_chain, quantities, path_init, sampler, vec_β_init
    # return model, sampler, model_chain, quantities, path_init, ψs
    # return model, sampler, path_init
end

function discrete_sampler(nv, n_donors, β_init, θ_init, ω_init, X, interventions)
    return MH(
        :β => DAGProposal(DAG(size(β_init, 1)), X, interventions),
        :θ => RandomWalkProposal(MvNormal(zeros(length(θ_init)), I * 0.03)),
        :ω => RandomWalkProposal(MvNormal(zeros(length(ω_init)), I * 0.03))
        # :σₓ => RandomWalkProposal(Normal(0., 0.10))
    )

end

function daggify(W::Matrix{T}) where T<:AbstractFloat
    dag = true 
    β = deepcopy(W)
    # println("initial matrix = $W")
    # threshold = minimum(β[abs.(β) .> 0]) # initialize threshold at min non-zero element
    sorted_vals = sort(abs.(vec(β[abs.(β) .> 0])))
    count = 1
    threshold = sorted_vals[count]
    sr = last(eigvals(β .* β))
    # sr = opnorm(β .* β, 2)
    if (typeof(sr) <: Complex) || (sr > .005)
        dag = false
    end

    while !dag 
        count += 1
        β[abs.(β) .<= threshold] .= 0.0
        sr = last(eigvals(β .* β))
        # sr = opnorm(β .* β, 2)
        if (typeof(sr) <: Real) && (sr < .005) && !isnothing(topological_sort(β .!= 0))
            dag = true
        else 
            threshold = sorted_vals[count]
        end
        # println("threshold = $threshold, β = $β, sr = $sr")
    end
    
    println("Converted to DAG after $count iterations")
    return β
end


# parse_chain(m[1], m[2], targets)
function parse_chain(chains::Chains, posterior_adjacency::Matrix{Matrix{Float64}}; targets=setdiff(ko_targets(), ko_controls()))

    n_samples = size(posterior_adjacency, 1) # slightly misleading - total mcmc samples is n_samples * n_chains
    n_chains  = size(posterior_adjacency, 2)

    @assert n_samples >= 50
    @assert n_chains  >= 1

    posterior_mean = mean(mean(posterior_adjacency; dims = 2)) # inner mean takes mean across chains, outer averages across samples
    nv = size(posterior_mean, 1)

    @assert nv == size(posterior_mean, 2)

    result = DataFrame() # convert to 'long' format

    for i in 1:nv
        for j in 1:nv

            # row_label = "gene_$i"
            # col_label = "gene_$j"
            row_label = targets[i]
            col_label = targets[j]

            if i == j

                row = (;
                    :row => row_label,
                    :col => col_label,
                    :estimate => posterior_mean[i, j],
                    Symbol("2.5%") => 0.0,
                    Symbol("97.5%") => 0.0,
                    :rhat => 1.0,
                    :ess => Float64(n_samples * n_chains),
                    :std => 0.0
                )

            else
                alt_i = i - Int(i > j)
                alt_j = j
                summary   = summarize(chains[[Symbol("β[$alt_i,$alt_j]")]])
                quantiles = quantile(chains[[Symbol("β[$alt_i,$alt_j]")]])

                row = (;
                    :row => row_label,
                    :col => col_label,
                    :estimate => posterior_mean[i, j],
                    Symbol("2.5%") => quantiles[:, Symbol("2.5%")],
                    Symbol("97.5%") => quantiles[:, Symbol("97.5%")],
                    :rhat => summary[:, :rhat],
                    :ess => summary[:, :ess],
                    :std => summary[:, :std]
                )

            end

            push!(result, row)
        end
    end

    return result
end

function tabulate_permutations(posterior_adjacency::Matrix{Matrix{Float64}})

    permutation_map = Vector{Int64}[]

    stats = Dict()

    n_samples = size(posterior_adjacency, 1)
    n_chains = size(posterior_adjacency, 1)
    for c in 1:n_chains
        counts = Dict()
        for i in 1:n_samples
            perm = topological_sort(posterior_adjacency[i, c] .!= 0)
            if length(permutation_map) == 0 || sum(permutation_map .== perm) == 0
                push!(permutation_map, perm)
            end

            println("perm = $perm, permutation_map = $permutation_map")
            idx = [i for i in 1:length(permutation_map) if permutation_map[i] == perm]
            if !(idx in keys(counts))
                push!(counts, idx => 0)
            end

            counts[idx] += 1 
        end
        stats[c] = (permutation_map, counts)
    end

    return stats
end

function extract_pair(chains::Chains, gene1::String, gene2::String; targets=setdiff(ko_targets(), ko_controls()))
    
    i = findall(x -> x == gene1, targets)[1]
    j = findall(x -> x == gene2, targets)[1]
    alt_i = i - Int(i > j)
    alt_j = j
    sym_forward   = Symbol("β[$alt_i,$alt_j]")
    i = findall(x -> x == gene2, targets)[1]
    j = findall(x -> x == gene1, targets)[1]
    alt_i = i - Int(i > j)
    alt_j = j
    sym_backward   = Symbol("β[$alt_i,$alt_j]")
    sub_chain   = chains[[sym_forward, sym_backward]]

    return sub_chain
end

function convert_to_reduced_adjacency(W::Matrix{T}) where T <: Real
    nv = size(W, 2)
    β = Array{T}(undef, nv - 1, nv) 

    for i in 1:nv
        for j in 1:nv
            if i == j
                continue
            else
                β[i - Int(i > j), j] = W[i, j]
            end
        end
    end

    return β
end

function convert_to_full_adjacency(β::Matrix{T}) where T <: Real
    nv = size(β, 2)
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

    return W
end
