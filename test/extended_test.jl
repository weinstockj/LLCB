using Distributions
using LinearAlgebra: I
using InferCausalGraph: d_connected_advi, parse_symbol_map
using Statistics: var, std, cor, cov
using Formatting: sprintf1
using DataFrames: DataFrame
using CSV: write
using Dates: now
using Turing: setprogress!


setprogress!(false)

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

function parse_sim_result(parsed_map, θ, x, y, Rsq)
    return (
        beta_x = parsed_map.beta_x,
        true_x = θ[1],
        beta_x_error = (parsed_map.beta_x - θ[1]) / θ[1],
        beta_x_interact_i = parsed_map.beta_x_interact_i,
        beta_x_interact_y = parsed_map.beta_x_interact_y,
        beta_pip = parsed_map.beta_pip,
        beta_pip_mean = parsed_map.beta_pip_mean,
        beta_pip_old = parsed_map.beta_pip_old,
        beta_pip_crude = parsed_map.beta_pip_crude,
        lambda_x = parsed_map.lambda_x,
        tau = parsed_map.tau,
        intercept = parsed_map.intercept,
        log_variance_intercept = parsed_map.log_variance_intercept,
        log_variance_beta_x_interact_i = parsed_map.log_variance_beta_x_interact_i,
        log_variance_beta_x_interact_y = parsed_map.log_variance_beta_x_interact_y,
        std_x = std(x),
        std_y = std(y),
        Rsq = Rsq,
        pearson_cor_x_y = cor(x, y),
        pearson_cor_pvalue = cortest(x, y)
    )
end

function apply_KO!(int::Vector{Float64}, vec::Vector{Float64}, ko_value = -3.0)
    for i in 1:length(int)
        if int[i] > 0
            vec[i] = ko_value 
        end
    end
end

function sim(include_z = false)

    N = 800
    P = 5

    x = rand(Normal(0.0, 1.0), N)
    z = rand(Normal(0.0, 1.0), N * (P - 1))
    z = reshape(z, N, P - 1)

    # true weights
    # X -> Y
    θ = rand(Normal(0.0, 3.0), P)
    θ[1] = -3.0 # fix first weight

    # simulate KO effects
    int_x = Float64.(rand(Bernoulli(0.2), N))
    int_y = Float64.(rand(Bernoulli(0.2), N))
    x_int_y = Vector{Float64}(undef, N)

    ko_value = -3.0

    # for i in 1:N
    #     if int_x[i] > 0
    #         x[i] = 0.0
    #     end
    # end
    #
    apply_KO!(int_x, x, ko_value)
    @assert sum(x .== ko_value) == sum(int_x)

    if include_z
        μ = x .* θ[1] + z * θ[2:P]
    else 
        μ = x .* θ[1]
    end

    # μ = ones(N) # condition on μ
    σ = 1.0
    var_μ = var(μ)
    y = vec(rand(MvNormal(μ, σ * I), 1))
    Rsq = var_μ / (var(y))

    # println(sprintf1("Simulated Var(Y) before KO is %0.2f", var(y)))
    # println(sprintf1("Simulated Rsquared is %0.2f", Rsq))
    # println(sprintf1("Simulated E(μ^2) is %0.2f", mean(μ .^ 2)))
    # println(sprintf1("Simulated E(μ) is %0.2f", mean(μ)))
    # println(sprintf1("Simulated Cov(X, Y) = sd(μ) * sd(x) before KO is %0.2f", cov(x, y)))

    for i in 1:N
        if int_y[i] > 0
            y[i] = ko_value
            x_int_y[i] = x[i]
        else
            x_int_y[i] = 0.0
        end
    end

    # println(sprintf1("Simulated E(Y)  is %0.2f", mean(y)))
    # println(sprintf1("Simulated Var(Y) is %0.2f", var(y)))
    # println(sprintf1("Simulated Var(X) is %0.2f", var(x)))
    # println(sprintf1("Simulated Cov(X, Y) = sd(μ) * sd(x) after KO is %0.2f", cov(x, y)))
    # println(sprintf1("Simulated Cov(μ, Y) after KO is %0.2f", cov(μ, y)))
    # println(sprintf1("Simulated Rsquared Y~μ is %0.2f", cor(μ, y) ^ 2))
    # println(sprintf1("Simulated Rsquared Y~X is %0.2f", cor(x, y) ^ 2))
    # println(sprintf1("Simulated Var(μ) is %0.2f", var_μ))

    int = hcat(int_x, x_int_y)

    model, sym2range = d_connected_advi(x, y, z, int)
    parsed_map = parse_symbol_map(model, sym2range)

    return parse_sim_result(parsed_map, θ, x, y, Rsq)
end

function sim_reverse(include_z = false)

    N = 800
    P = 5

    y = rand(Normal(0.0, 1.0), N)
    z = rand(Normal(0.0, 1.0), N * (P - 1))
    z = reshape(z, N, P - 1)

    # true weights
    # Y -> X
    θ = rand(Normal(0.0, 3.0), P)
    θ[1] = -3.0 # fix first weight

    # simulate KO effects
    int_x = Float64.(rand(Bernoulli(0.2), N))
    int_y = Float64.(rand(Bernoulli(0.2), N))
    x_int_y = Vector{Float64}(undef, N)

    ko_value = -3.0

    apply_KO!(int_y, y, ko_value)
    @assert sum(y .== ko_value) == sum(int_y)
    # for i in 1:N
    #     if int_y[i] > 0
    #         y[i] = 0.0
    #     end
    # end

    if include_z
        μ = y .* θ[1] + z * θ[2:P]
    else 
        μ = y .* θ[1]
    end

    σ = 1.0
    var_μ = var(μ)
    x = vec(rand(MvNormal(μ, σ * I), 1))

    Rsq = var_μ / (var(x))

    attentuated_c = (θ[1] / 2.94 ^ 2) / (1 + σ / var_μ)

    # println(sprintf1("Simulated Rsquared X~μ is %0.2f", Rsq))

    apply_KO!(int_x, x, ko_value)
    @assert sum(x .== ko_value) == sum(int_x)
    # for i in 1:N
    #     if int_x[i] > 0
    #         x[i] = 0.0
    #     end
    # end


    for i in 1:N
        if int_y[i] > 0
            x_int_y[i] = x[i]
        else
            x_int_y[i] = 0.0
        end
    end

    # println(sprintf1("Simulated Rsquared Y~μ is %0.2f", cor(μ, y) ^ 2))
    # println(sprintf1("Simulated Rsquared Y~X is %0.2f", cor(x, y) ^ 2))
    # println(sprintf1("Simulated var_μ of coefficient is %0.2f", var_μ))
    # println(sprintf1("Simulated attentuation of coefficient is %0.2f", 1 + σ / var_μ))

    # x = zscore(x)
    # y = zscore(y)
    # println(sprintf1("Simulated std(x) is %0.2f", std(x)))
    # println(sprintf1("Simulated std(y) is %0.2f", std(y)))

    int = hcat(int_x, x_int_y)

    # println("int: = ", int)

    model, sym2range = d_connected_advi(x, y, z, int)
    parsed_map = parse_symbol_map(model, sym2range)

    return parse_sim_result(parsed_map, θ, x, y, Rsq)
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
