using Distributions
using LinearAlgebra: I
# using InferCausalGraph: d_connected_advi, parse_symbol_map
using InferCausalGraph: interventionGraph, fit_model, get_model_params, get_sampling_params
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
    model_pars = get_model_params(false, .035, .01)
    sampling_pars = get_sampling_params(true)
    model = fit_model(graph, true, model_pars, sampling_pars)
    return model, graph
end

function sim_multi_modal_expression_and_fit_model()
    # true_grn = read_true_grn() .* .025
    #
    true_grn_a = tmp_grn() * 40.0
    true_grn_b = tmp_grn_alt() * 40.0
    # true_grn = big_tmp_grn()
    # expression = sim_expression(true_grn, 3, 250) 
    expression_a = sim_expression(true_grn_a, 3, 50) 
    expression_b = sim_expression(true_grn_b, 3, 50) 
    expression = vcat(expression_a, expression_b)
    graph = interventionGraph(expression)
    model_pars = get_model_params(false, .035, .01)
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

function sim_expression_not_topoligcal(true_adjacency::Matrix{Float64}, n_donors::Int64 = 3, n_replicates_per_donor::Int64 = 150)
    nv = size(true_adjacency, 1)
    # n_donors = 3
    # n_replicates_per_donor = 1
    θ = rand(Normal(0, .01), nv)
    ω = rand(Normal(0, .1), n_donors)
    μ = 4

    x = zeros(nv * n_donors * n_replicates_per_donor, nv)
    dampen = 1.0
    βtrue = true_adjacency .* dampen

    intervention_vec = Vector{String}(undef, nv * n_donors * n_replicates_per_donor)
    donor_vec = Vector{String}(undef, nv * n_donors * n_replicates_per_donor)

    x_init = rand.(Poisson.(exp.(θ .+ μ)), 1) 
    x_init = vcat(x_init...) # nv x 1
    # x_init = transpose(x_init) # 1 x nv

    # println("x_init")
    # show(x_init)

    i = 1

    for j in 1:nv # KO index
        adjacency = deepcopy(true_adjacency .!= 0) # binary

        adjacency[:, j] .= 0 # remove parents
        adjacency[j, :] .= 0 # remove children because expression is 0
        gene = "gene_$(j)"

        β = βtrue .* adjacency

        xᵢ = deepcopy(x_init)
        η = zeros(1, nv)
        # println("β = $(show(β)), j = $j")
        # if rand(Bernoulli(.7), 1)[1]
        # end

        for d in 1:n_donors
            for z in 1:n_replicates_per_donor
                # η = x_init * β 
                # η = hcat(η...)
                for z in 1:nv
                    η[z] = sum(β[:, z] .* x_init)
                end

                # println("η[1, 1] = $(η[1, 1])")
                # η = η .+ transpose(θ) .+ μ .+ ω[d]
                η = η .+ μ .+ ω[d]
                # println("η[1, 1] = $(η[1, 1])")
                # η = η .+ ω[d]
                xᵢ = rand.(LogNormal.(η, .08), 1)
                xᵢ = hcat(xᵢ...)
                # println("xᵢ[1, 1] = $(xᵢ[1, 1])")
                
                xᵢ[1, j] = 0.0
                @assert size(xᵢ, 1) == 1
                @assert size(xᵢ, 2) == nv
                x[i, :] = xᵢ[1, :]
                intervention_vec[i] = gene
                donor_vec[i] = string(d)
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

# function parse_sim_result(parsed_map, θ, x, y, Rsq)
#     return (
#         beta_x = parsed_map.beta_x,
#         true_x = θ[1],
#         beta_x_error = (parsed_map.beta_x - θ[1]) / θ[1],
#         beta_x_interact_i = parsed_map.beta_x_interact_i,
#         beta_x_interact_y = parsed_map.beta_x_interact_y,
#         beta_pip = parsed_map.beta_pip,
#         beta_pip_mean = parsed_map.beta_pip_mean,
#         beta_pip_old = parsed_map.beta_pip_old,
#         beta_pip_crude = parsed_map.beta_pip_crude,
#         lambda_x = parsed_map.lambda_x,
#         tau = parsed_map.tau,
#         intercept = parsed_map.intercept,
#         log_variance_intercept = parsed_map.log_variance_intercept,
#         log_variance_beta_x_interact_i = parsed_map.log_variance_beta_x_interact_i,
#         log_variance_beta_x_interact_y = parsed_map.log_variance_beta_x_interact_y,
#         std_x = std(x),
#         std_y = std(y),
#         Rsq = Rsq,
#         pearson_cor_x_y = cor(x, y),
#         pearson_cor_pvalue = cortest(x, y)
#     )
# end
#
# function apply_KO!(int::Vector{Float64}, vec::Vector{Float64}, ko_value = -3.0)
#     for i in 1:length(int)
#         if int[i] > 0
#             vec[i] = ko_value 
#         end
#     end
# end
#
# function sim(include_z = false)
#
#     N = 800
#     P = 5
#
#     x = rand(Normal(0.0, 1.0), N)
#     z = rand(Normal(0.0, 1.0), N * (P - 1))
#     z = reshape(z, N, P - 1)
#
#     # true weights
#     # X -> Y
#     θ = rand(Normal(0.0, 3.0), P)
#     θ[1] = -3.0 # fix first weight
#
#     # simulate KO effects
#     int_x = Float64.(rand(Bernoulli(0.2), N))
#     int_y = Float64.(rand(Bernoulli(0.2), N))
#     x_int_y = Vector{Float64}(undef, N)
#
#     ko_value = -3.0
#
#     # for i in 1:N
#     #     if int_x[i] > 0
#     #         x[i] = 0.0
#     #     end
#     # end
#     #
#     apply_KO!(int_x, x, ko_value)
#     @assert sum(x .== ko_value) == sum(int_x)
#
#     if include_z
#         μ = x .* θ[1] + z * θ[2:P]
#     else 
#         μ = x .* θ[1]
#     end
#
#     # μ = ones(N) # condition on μ
#     σ = 1.0
#     var_μ = var(μ)
#     y = vec(rand(MvNormal(μ, σ * I), 1))
#     Rsq = var_μ / (var(y))
#
#     # println(sprintf1("Simulated Var(Y) before KO is %0.2f", var(y)))
#     # println(sprintf1("Simulated Rsquared is %0.2f", Rsq))
#     # println(sprintf1("Simulated E(μ^2) is %0.2f", mean(μ .^ 2)))
#     # println(sprintf1("Simulated E(μ) is %0.2f", mean(μ)))
#     # println(sprintf1("Simulated Cov(X, Y) = sd(μ) * sd(x) before KO is %0.2f", cov(x, y)))
#
#     for i in 1:N
#         if int_y[i] > 0
#             y[i] = ko_value
#             x_int_y[i] = x[i]
#         else
#             x_int_y[i] = 0.0
#         end
#     end
#
#     # println(sprintf1("Simulated E(Y)  is %0.2f", mean(y)))
#     # println(sprintf1("Simulated Var(Y) is %0.2f", var(y)))
#     # println(sprintf1("Simulated Var(X) is %0.2f", var(x)))
#     # println(sprintf1("Simulated Cov(X, Y) = sd(μ) * sd(x) after KO is %0.2f", cov(x, y)))
#     # println(sprintf1("Simulated Cov(μ, Y) after KO is %0.2f", cov(μ, y)))
#     # println(sprintf1("Simulated Rsquared Y~μ is %0.2f", cor(μ, y) ^ 2))
#     # println(sprintf1("Simulated Rsquared Y~X is %0.2f", cor(x, y) ^ 2))
#     # println(sprintf1("Simulated Var(μ) is %0.2f", var_μ))
#
#     int = hcat(int_x, x_int_y)
#
#     model, sym2range = d_connected_advi(x, y, z, int)
#     parsed_map = parse_symbol_map(model, sym2range)
#
#     return parse_sim_result(parsed_map, θ, x, y, Rsq)
# end
#
# function sim_reverse(include_z = false)
#
#     N = 800
#     P = 5
#
#     y = rand(Normal(0.0, 1.0), N)
#     z = rand(Normal(0.0, 1.0), N * (P - 1))
#     z = reshape(z, N, P - 1)
#
#     # true weights
#     # Y -> X
#     θ = rand(Normal(0.0, 3.0), P)
#     θ[1] = -3.0 # fix first weight
#
#     # simulate KO effects
#     int_x = Float64.(rand(Bernoulli(0.2), N))
#     int_y = Float64.(rand(Bernoulli(0.2), N))
#     x_int_y = Vector{Float64}(undef, N)
#
#     ko_value = -3.0
#
#     apply_KO!(int_y, y, ko_value)
#     @assert sum(y .== ko_value) == sum(int_y)
#     # for i in 1:N
#     #     if int_y[i] > 0
#     #         y[i] = 0.0
#     #     end
#     # end
#
#     if include_z
#         μ = y .* θ[1] + z * θ[2:P]
#     else 
#         μ = y .* θ[1]
#     end
#
#     σ = 1.0
#     var_μ = var(μ)
#     x = vec(rand(MvNormal(μ, σ * I), 1))
#
#     Rsq = var_μ / (var(x))
#
#     attentuated_c = (θ[1] / 2.94 ^ 2) / (1 + σ / var_μ)
#
#     # println(sprintf1("Simulated Rsquared X~μ is %0.2f", Rsq))
#
#     apply_KO!(int_x, x, ko_value)
#     @assert sum(x .== ko_value) == sum(int_x)
#     # for i in 1:N
#     #     if int_x[i] > 0
#     #         x[i] = 0.0
#     #     end
#     # end
#
#
#     for i in 1:N
#         if int_y[i] > 0
#             x_int_y[i] = x[i]
#         else
#             x_int_y[i] = 0.0
#         end
#     end
#
#     # println(sprintf1("Simulated Rsquared Y~μ is %0.2f", cor(μ, y) ^ 2))
#     # println(sprintf1("Simulated Rsquared Y~X is %0.2f", cor(x, y) ^ 2))
#     # println(sprintf1("Simulated var_μ of coefficient is %0.2f", var_μ))
#     # println(sprintf1("Simulated attentuation of coefficient is %0.2f", 1 + σ / var_μ))
#
#     # x = zscore(x)
#     # y = zscore(y)
#     # println(sprintf1("Simulated std(x) is %0.2f", std(x)))
#     # println(sprintf1("Simulated std(y) is %0.2f", std(y)))
#
#     int = hcat(int_x, x_int_y)
#
#     # println("int: = ", int)
#
#     model, sym2range = d_connected_advi(x, y, z, int)
#     parsed_map = parse_symbol_map(model, sym2range)
#
#     return parse_sim_result(parsed_map, θ, x, y, Rsq)
# end

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
