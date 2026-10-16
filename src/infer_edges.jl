Turing.setprogress!(true)

"""
    get_model_params(simulation, l1_penalty = 0.01, R_scale = 0.05) -> NamedTuple

Prior scales for the published model ([`fit_cyclic_model_legacy`](@ref)),
as used in the manuscript. `simulation = true` selects the values used for the
simulation study; `false` selects those for the real data, where `l1_penalty`
(scale of the Laplace prior on the direct effects; smaller means more
shrinkage) and `R_scale` can be overridden.

Fields: `β_scale`, `R_scale`, `l1_penalty`, `l2_penalty`, `β_sum_scale`,
`θ_scale`, `ω_scale`, `σₓ_scale`. The positional `fit_cyclic_model` method maps
`l1_penalty` to `prior_scale` and ignores the rest.
"""
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

"""
    get_sampling_params(simulation) -> NamedTuple

Sampler settings for the published model: `warmups`, `acceptance`,
`max_depth`, `init_ϵ` and `samples`. `simulation = true` selects the lighter
settings used for the simulation study.
"""
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

# `fit_model` and `discrete_sampler` (the discrete-DAG MH sampler) were removed
# in v0.3.0. They depended on `DAG`, `DAGProposal`, `DAGScorer` and `bge`,
# defined in DAGsampler.jl / DAGproposal.jl, which were not part of the v0.2.0b
# module build or the published analysis. See tag v0.2.0b for the source.

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
function parse_chain(chains::MCMCChains.Chains, posterior_adjacency::AbstractMatrix; targets=setdiff(ko_targets(), ko_controls()))

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

function tabulate_permutations(posterior_adjacency::AbstractMatrix)

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

function extract_pair(chains::MCMCChains.Chains, gene1::String, gene2::String; targets=setdiff(ko_targets(), ko_controls()))
    
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
