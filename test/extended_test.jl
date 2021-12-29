using Distributions
using LinearAlgebra: I
using InferCausalGraph: d_connected_advi, parse_symbol_map
using Statistics: var
using Formatting: sprintf1
using DataFrames: DataFrame
using CSV: write

function sim()

    N = 800
    P = 5

    x = rand(Normal(0.0, 1.0), N * P)
    x = reshape(x, N, P)

    # true weights
    # X -> Y
    θ = rand(Normal(0.0, 3.0), P)
    θ[1] = -3.0 # fix first weight

    # simulate KO effects
    int_x = Float64.(rand(Bernoulli(0.2), N))
    int_y = Float64.(rand(Bernoulli(0.2), N))

    for i in 1:N
        if int_x[i] > 0
            x[i, 1] = 0.0
        end
    end

    int = hcat(int_x, int_y)

    μ = x * θ
    σ = 1.0

    var_μ = var(μ)
    y = vec(rand(MvNormal(μ, σ * I), 1))

    Rsq = var_μ / (var(y))

    println(sprintf1("Simulated Rsquared is %0.2f", Rsq))

    for i in 1:N
        if int_y[i] > 0
            y[i] = 0.0
        end
    end

    z = x[:, 2:P]

    model, sym2range = d_connected_advi(x[:, 1], y, z, int)
    parsed_map = parse_symbol_map(model, sym2range)

    # println("θ = $(θ)")
    println("parsed_map = $(parsed_map)")

    return (
        beta_x = parsed_map.beta_x,
        true_x = θ[1],
        beta_x_error = (parsed_map.beta_x - θ[1]) / θ[1],
        beta_x_interact_i = parsed_map.beta_x_interact_i,
        beta_pip = parsed_map.beta_pip,
        beta_pip_old = parsed_map.beta_pip_old,
        log_variance_intercept = parsed_map.log_variance_intercept,
        log_variance_beta_x_interact_i = parsed_map.log_variance_beta_x_interact_i,
        log_variance_beta_x_interact_y = parsed_map.log_variance_beta_x_interact_y
    )
end

function test_out_dir()

    out_dir = "/oak/stanford/groups/pritch/users/jweinstk/network_inference/InferCausalGraph/output/simulation_output"
    return out_dir
end

function write_sims()
    N_SIMS = 100 # 462 seconds
    sims = [sim() for i in 1:N_SIMS]
    df = DataFrame(sims)
    
    write(joinpath(test_out_dir(), "simulation.csv"), df)
    return df
end
