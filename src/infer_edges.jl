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
