using Distributions
using LinearAlgebra: I
# using InferCausalGraph: d_connected_advi, parse_symbol_map
using InferCausalGraph: interventionGraph, fit_model, get_model_params, get_sampling_params, DAGScorer, bge
using Statistics: var, std, cor, cov
using Formatting: sprintf1
using DataFrames: DataFrame, rename!, vcat
using CSV: write, read
using Dates: now
using Turing: setprogress!


setprogress!(true)

function cortest(x::Vector{Float64}, y::Vector{Float64})
    if length(x) == length(y)
        return 2 * ccdf(Normal(), atanh(abs(cor(x, y))) * sqrt(length(x) - 3))
    else
        error("x and y have different lengths")
    end
end

function zscore(x)
    μ = mean(x)
    σ = std(x)
    return (x .- μ) ./ σ
end

function read_true_grn(nv::Int64 = 5)
    path = "/oak/stanford/groups/pritch/users/jweinstk/network_inference/ground_truth_simulation/output/true_grn.csv"
    names = ["gene_$i" for i in 1:24]
    grn = Matrix(read(path, header = names, DataFrame))

    return grn[1:nv, 1:nv]
end

function tmp_grn()
    true_grn = zeros(5, 5)
    true_grn[3, 1] = 0.009 #index β[2,1]
    true_grn[3, 2] = 0.007 #index β[2,2]
    # true_grn[3, 5] = 0.07 #index β[3,5]
    true_grn[1, 4] = -0.006 #index β[1,4]
    true_grn[2, 5] = -0.008 #index β[2,5]
    return true_grn
end

function chain_graph()
    true_grn = zeros(5, 5)
    true_grn[1, 2] = 0.5 
    true_grn[2, 3] = 0.5 
    true_grn[3, 4] = 0.5 
    true_grn[4, 5] = 0.5 
    return true_grn
end

function chain_graph_alt()
    true_grn = zeros(5, 5)
    true_grn[1, 2] = 0.5 
    true_grn[2, 3] = 0.5 
    true_grn[3, 5] = 0.5 
    true_grn[5, 4] = 0.5 
    return true_grn
end


function tmp_grn_alt()
    true_grn = zeros(5, 5)
    true_grn[3, 1] = 0.009 #index β[2,1]
    true_grn[3, 2] = 0.007 #index β[2,2]
    # true_grn[3, 5] = 0.07 #index β[3,5]
    true_grn[1, 4] = -0.006 #index β[1,4]
    true_grn[5, 2] = -0.008 #index β[2,5]
    return true_grn
end

function big_tmp_grn()
    true_grn = zeros(10, 10)
    true_grn[3, 1] = 0.19 #index 
    true_grn[3, 2] = 0.13 #index 
    true_grn[8, 3] = 0.17 #index 
    true_grn[1, 4] = -0.33 #index 
    true_grn[1, 7] = -0.28 #index 
    true_grn[2, 5] = 0.22 #index 
    # true_grn[2, 8] = -0.18 #index 
    true_grn[2, 9] = 0.12 #index 
    true_grn[8, 6] = 0.12 #index 
    true_grn[9, 4] = 0.12 #index 
    true_grn[10, 5] = 0.17 #index 
    true_grn[10, 6] = -0.22 #index 
    return true_grn
end

function sim_expression_and_fit_model()
    # true_grn = read_true_grn() .* .025
    #
    true_grn = tmp_grn() * 40.0
    # true_grn = big_tmp_grn()
    # expression = sim_expression(true_grn, 3, 250) 
    expression = sim_expression(true_grn, 3, 50) 
    graph = interventionGraph(expression)
    # model_pars = get_model_params(false, .035, .01)
    model_pars = get_model_params(false, .01, .01)
    sampling_pars = get_sampling_params(true)
    model = fit_model(graph, true, model_pars, sampling_pars)
    return model, graph
end

function sim_multi_modal_expression_and_fit_model()
    #
    # true_grn_a = tmp_grn() * 40.0
    # true_grn_b = tmp_grn_alt() * 40.0
    true_grn_a = chain_graph()
    true_grn_b = chain_graph_alt()
    # expression = sim_expression(true_grn, 3, 250) 
    expression_a = sim_expression(true_grn_a, 3, 50) 
    expression_b = sim_expression(true_grn_b, 3, 50) 
    expression = vcat(expression_a, expression_b)
    graph = interventionGraph(expression)
    model_pars = get_model_params(false, .01, .01)
    sampling_pars = get_sampling_params(true)
    model = fit_model(graph, true, model_pars, sampling_pars)
    return model, graph
end

function sim_expression(true_adjacency::Matrix{Float64}, n_donors::Int64 = 3, n_replicates_per_donor::Int64 = 50, verbose=false)
    nv = size(true_adjacency, 1)
    θ = rand(Normal(0, .1), nv)
    ω = rand(Normal(0, .1), n_donors)
    μ = 5.0

    x = zeros(nv * n_donors * n_replicates_per_donor, nv)
    intervention_vec = Vector{String}(undef, nv * n_donors * n_replicates_per_donor)
    donor_vec = Vector{String}(undef, nv * n_donors * n_replicates_per_donor)
    σₓ = 0.03

    i = 1

    for k in 1:nv
        β = deepcopy(true_adjacency)
        β[:, k] .= 0
        nodes_without_parents = Vector{Int}()
        for l in 1:nv
            if all(sum(β[:, l]) == 0)
                push!(nodes_without_parents, l)
            end
        end
        if verbose
            println("nodes_without_parents = $nodes_without_parents")
        end

        @assert length(nodes_without_parents) >= 1
        for d in 1:n_donors
            for z in 1:n_replicates_per_donor
                donor_vec[i] = string(d)
                intervention_vec[i] = "gene_$k"
                nodes_remaining = Vector{Int}()
                xᵢ = zeros(nv)
                for m in nodes_without_parents
                    push!(nodes_remaining, m)
                    xᵢ[m] = Float64(rand(Poisson(exp(μ + ω[d] + θ[m])), 1)[1])
                end

                xᵢ[k] = 0.0

                while length(nodes_remaining) >= 1
                    node = pop!(nodes_remaining)
                    children = findall(x -> x != 0, β[node, :])
                    for child in children

                        parents = findall(x -> x != 0, β[:, child])

                        if verbose
                            println("node = $node, child = $child, xᵢ = $xᵢ")
                            println("xᵢ[$(parents)]  = $(xᵢ[parents]), parents=$parents ")
                            println(xᵢ[setdiff(parents, k)])
                        end

                        if all(xᵢ[setdiff(parents, k)] .> 0) & (xᵢ[child] == 0.0)
                            # require all incoming edges to be set, and the child to not have been visited already
                            push!(nodes_remaining, child)
                            η = μ + θ[child] + ω[d] + sum(log1p.(xᵢ[parents]) .* β[parents, child])
                            xᵢ[child] = rand(LogNormal(η, σₓ), 1)[1]
                        else 
                            continue
                        end
                    end
                end

                x[i, :] = xᵢ
                i += 1
            end
        end
    end

    df = DataFrame(x, :auto)
    rename!(df, [Symbol("gene_$i") for i in 1:nv])
    df.intervention = intervention_vec
    df.donor = donor_vec

    return df
end

function test_bge()
    names = ["A", "B", "C", "D", "E", "F", "G"]
    path = "/home/users/jweinstk/.julia/dev/InferCausalGraph/test/bnlearn_gauss_net.csv"
    grn = Matrix(read(path, DataFrame; types = Dict(
        1 => Bool,
        2 => Bool,
        3 => Bool,
        4 => Bool,
        5 => Bool,
        6 => Bool,
        7 => Bool
    )))

    path = "/home/users/jweinstk/.julia/dev/InferCausalGraph/test/bnlearn_gauss_net_data.csv"
    data = Matrix(read(path, DataFrame; types = Dict(
        1 => Float64,
        2 => Float64,
        3 => Float64,
        4 => Float64,
        5 => Float64,
        6 => Float64,
        7 => Float64
    )))

    # score should be -53258.94 according to bnlearn, and -53137 according to BiDAG
    grn = BitMatrix(grn)
    scorer = DAGScorer(grn)
    val = bge(scorer, data)
    @assert abs(val - 53137.476 < 1.0)
    return grn, data
end

function test_out_dir()

    out_dir = "/oak/stanford/groups/pritch/users/jweinstk/network_inference/InferCausalGraph/output/simulation_output"
    return out_dir
end

function parallel_sim_df(x_cause_y, include_confounders)
    N_SIMS = 1000
    sims = Vector{NamedTuple}(undef, N_SIMS)
    Threads.@threads for i in 1:N_SIMS
        if x_cause_y
            sims[i] = sim(include_confounders)
        else
            sims[i] = sim_reverse(include_confounders)
        end
    end
    df = DataFrame(sims)
    return df
end

function write_sims()
    # 3040 seconds overall
    # N_SIMS = 1000 # 4.62 seconds per simulation
    #
    date = "2022_01_07"
    
    @info "$(now()) running x->y, no z"
    df = parallel_sim_df(true, false)
    write(joinpath(test_out_dir(), "simulation_x_cause_y_no_confounders_$(date).csv"), df)

    
    @info "$(now()) running x->y, with z"
    df = parallel_sim_df(true, true)
    write(joinpath(test_out_dir(), "simulation_x_cause_y_with_confounders_$(date).csv"), df)

    
    @info "$(now()) running y->x, no z"
    df = parallel_sim_df(false, false)
    write(joinpath(test_out_dir(), "simulation_y_cause_x_no_confounders_$(date).csv"), df)

    
    @info "$(now()) running y->x, with z"
    df = parallel_sim_df(false, true)
    write(joinpath(test_out_dir(), "simulation_y_cause_x_with_confounders_$(date).csv"), df)
end
