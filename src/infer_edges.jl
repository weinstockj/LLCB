Turing.setprogress!(false)
@model function bayes_linear(x, y, i, ::Type{T} = Float64) where {T}
    variance_prior ~ truncated(Normal(0, 3), 0, Inf)
    intercept ~ Normal(0.0, 10.0)

    n = size(x, 1)
    p_x = size(x, 2)
    p_i = size(i, 2)
    beta_x = Vector{T}(undef, p_x)
    tau = 0.10

    # lambda_x ~ filldist(truncated(Cauchy(0, 1), 0, Inf), p_x)
    lambda_x ~ filldist(truncated(TDist(5), 0, Inf), p_x) # half T with 5 df

    # beta .~ Laplace(0.0, 1.0)
    # beta ~ filldist(Laplace(0.0, 1.0), p)
    # for i in 1:p_x
        # beta_x[i] ~ Normal(0, lambda_x[i] * tau)
    # end
    beta_x .~ Normal.(0, lambda_x .* tau)
    # beta ~ MvNormal(p, 1.0)
    #
    #
    beta_i ~ filldist(Normal(0, 10.0), p_i)

    yhat = intercept .+ x * beta_x .+ i * beta_i
    # y ~ MvNormal(yhat, sqrt(variance_prior))
    # for i in 1:n
    #     y[i] ~ Normal(yhat[i], sqrt(variance_prior))
    # end
    y .~ Normal.(yhat, sqrt(variance_prior))
end

@model function bayes_linear_var(x, y, i, ::Type{T} = Float64) where {T}
    intercept ~ Normal(0.0, 10.0)
    log_variance_intercept ~ Normal(0.0, 1.0)

    n = size(x, 1)
    p_x = size(x, 2)
    p_i = size(i, 2)
    beta_x = Vector{T}(undef, p_x)
    # tau = 0.10
    tau ~ truncated(TDist(3), 0, Inf)

    # lambda_x ~ filldist(truncated(Cauchy(0, 1), 0, Inf), p_x)
    lambda_x ~ filldist(truncated(TDist(3), 0, Inf), p_x) # half T with 5 df

    beta_x .~ Normal.(0, lambda_x .* tau)
    beta_i ~ filldist(Normal(0, 10.0), p_i)

    yhat = intercept .+ x * beta_x .+ i * beta_i


    log_variance_beta_i ~ filldist(Normal(0.0, 1.0), p_i)
    log_variance_hat = log_variance_intercept .+ i * log_variance_beta_i
    y .~ Normal.(yhat, sqrt.(exp.(log_variance_hat)))
end

function d_connected_nuts(x::T, y::T, z::Matrix{Union{Missing, Float64}}) where T<:AbstractArray{Union{Missing, Float64}}
    full_model       = bayes_linear_lasso(y, hcat(x, z))
    restricted_model = bayes_linear_lasso(y, z) 

    full_model_chain = sample(full_model, NUTS(0.65), MCMCThreads(), 200, 4)
    # advi = ADVI(20, 100)
    # full_model_chain_vi = vi(full_model, advi; optimizer = Flux.ADAM())
    # full_model_chain_vi = vi(full_model, advi)
    # full_model_chain_vi = vi(full_model, advi; optimizer = DecayedADAGrad(1e-2, 1.1, 0.9))
    # @time restricted_model_chain = sample(restricted_model, NUTS(0.65), 1000)

    # return full_model_chain, restricted_model_chain
    # return full_model_chain_vi
    return full_model_chain
end

function d_connected_advi(x::T, y::T, z::Matrix{Float64}, i::Matrix{Float64}) where T<:AbstractArray{Float64}
    # Turing.setadbackend(:tracker)
    Turing.setadbackend(:forwarddiff)
    full_model       = bayes_linear_var(hcat(x, z), y, i)
    _, sym2range = bijector(full_model, Val(true));

    advi = ADVI(5, 1000)
    full_model_chain_vi = vi(full_model, advi)

    return full_model_chain_vi, sym2range
end

function d_connected_map(x::T, y::T, z::Matrix{Float64}) where T<:AbstractVector{Float64}
    full_model       = bayes_linear_lasso(y, hcat(x, z))
    restricted_model = bayes_linear_lasso(y, z) 
    #
    full_model_chain = optimize(full_model, MAP())
    # return full_model_chain, restricted_model_chain
    return full_model_chain
    # return full_model_chain
end

function test_me(data::DataFrame)
   x = @view data[!, "IRF4"]
   y = @view data[!, "IL2RA"]
   # z = Matrix(@view data[:, 20:30])
   z_cols = setdiff(names(data), ["IRF4", "IL2RA", "intervention"])
   z = Matrix(@view data[:, z_cols])
   # @time models_nuts = d_connected_nuts(x, y, z) # 1594 seconds
   @info "$(now()) now running advi"
   @time models_advi = d_connected_advi(x, y, z) # 30 seconds with 100 gradient setps
   @info "$(now()) now running MAP"
   @time models_map = d_connected_map(x, y, z)
   @info "$(now()) now done"
   return models_map, models_advi
end

function estimate_skeleton(data::DataFrame, load_from_cache = false)
    cols = setdiff(ko_targets(), ko_controls())
    n_cols = length(cols)
    n_rows = nrow(data)
    n_vals = Int64(n_cols * (n_cols - 1) / 2)

    dir = output_dir()

    if load_from_cache
        values = deserialize(joinpath(dir, "values.store"))
        names = deserialize(joinpath(dir, "names.store"))
        maps = deserialize(joinpath(dir, "maps.store"))
        return values, names, maps
    end

    values  = Vector{Bijectors.MultivariateTransformed}(undef, n_vals)
    names   = Vector{Vector{String}}(undef, n_vals)
    maps   = Vector{NamedTuple}(undef, n_vals)

    @info "$(now()) Now estimating skeleton"
    # takes 1475 seconds
    # takes 393 seconds with 24 genes and 12 threads
    # takes 140 seconds with laplace prior and 12 threads
    # takes 655 seconds with horseshoe prior and 12 threads
    # takes 685 seconds with horseshoe prior and 24 threads
    # takes 695 seconds with horseshoe prior and 24 threads + variance terms

    progress_bar = Progress(n_vals, 1, "Estimating edge links...", 50)
    Threads.@threads for i in 1:n_cols
    # for i in 1:n_cols
        y_name  = cols[i]
        y = @view data[!, y_name]
        # @info "y_name is now $y_name, i = $i"
        # y = data[!, y_name]
        for j in 1:i
            x_name  = cols[j]
            # @info "x_name is now $x_name, j = $j"
            if j == i 
                # @info "$(now()) skipping x_name = $x_name y_name = $y_name"
                continue
            end
            x_interact_i = Vector{Float64}(undef, n_rows)
            x_interact_y = Vector{Float64}(undef, n_rows)
            z_names = setdiff(cols, [x_name, y_name])
            x = @view data[!, x_name]
            z = Matrix(@view data[:, z_names])
            for r in 1:n_rows
                if data.intervention[r] != x_name
                    x_interact_i[r] = 0.0
                else 
                    x_interact_i[r] = 1.0
                end
                if data.intervention[r] != y_name
                    x_interact_y[r] = 0.0
                else 
                    x_interact_y[r] = x[r] 
                end
            end
            model, sym2range = d_connected_advi(x, y, z, hcat(x_interact_i, x_interact_y))
            index = j + sum(collect(1:(i-2)))
            values[index] = model 
            names[index] = [x_name, y_name]
            maps[index] = sym2range
            next!(progress_bar)
        end
    end

    @info "$(now()) Done estimating skeleton"


    @info "$(now()) Now serializing"

    serialize(joinpath(dir, "values.store"), values)
    serialize(joinpath(dir, "names.store"), names)
    serialize(joinpath(dir, "maps.store"), maps)

    return values, names, maps
end

function parse_symbol_map(posterior_vi::Bijectors.MultivariateTransformed, sym2range::NamedTuple)
    @assert length(sym2range) >= 5

    n_sims = 1000
    samples = rand(posterior_vi, n_sims) # sample from posterior
    mean_samples = mean(samples, dims = 2) # means of columns - posterior expectation estimates

    beta_x_index = sym2range.beta_x[1][1]  # e.g., first index is 26:26, second index is 26
    lambda_x_index = sym2range.lambda_x[1][1]
    beta_x_interact_i_index = sym2range.beta_i[1][1]
    beta_x_interact_y_index = sym2range.beta_i[1][2]
    log_variance_intercept_index = sym2range.log_variance_intercept[1][1]
    log_variance_beta_x_interact_i_index = sym2range.log_variance_beta_i[1][1]
    log_variance_beta_x_interact_y_index = sym2range.log_variance_beta_i[1][2]
    intercept_index = sym2range.intercept[1][1]
    tau_index = sym2range.tau[1][1]

    κ = mean(1.0 ./ (1.0 .+ samples[lambda_x_index] .^ 2 .* samples[tau_index] .^ 2))

    lambda_pip_threshold = 5.0

    return (
        beta_x                         = mean_samples[beta_x_index],
        beta_pip_old                   = mean(samples[lambda_x_index, :] .> lambda_pip_threshold),
        beta_pip                       = 1 - κ,
        lambda_x                       = mean_samples[lambda_x_index],
        tau                            = mean_samples[tau_index],
        κ                              = κ,
        beta_x_interact_i              = mean_samples[beta_x_interact_i_index],
        beta_x_interact_y              = mean_samples[beta_x_interact_y_index],
        log_variance_intercept         = mean_samples[log_variance_intercept_index],
        log_variance_beta_x_interact_i = mean_samples[log_variance_beta_x_interact_i_index],
        log_variance_beta_x_interact_y = mean_samples[log_variance_beta_x_interact_y_index],
        intercept                      = mean_samples[intercept_index]
    )
end

function parse_skeleton(posterior::Vector{Bijectors.MultivariateTransformed}, names::Vector{Vector{String}}, maps::Vector{NamedTuple})
    @assert length(posterior) == length(names)
    @assert length(posterior) == length(maps)

    result = DataFrame()

    for i in 1:length(posterior)
        parsed_map = parse_symbol_map(posterior[i], maps[i])
        labeled_parsed_map = (; parsed_map..., x = names[i][1], y = names[i][2]) 
        push!(result, labeled_parsed_map)
    end

    return result
end

function posterior_predictive(posterior::Bijectors.MultivariateTransformed, names::Vector{String}, sym2range::NamedTuple, data::DataFrame)

    cols = setdiff(ko_targets(), ko_controls())
    x_name = names[1]
    y_name = names[2]
    z_names = setdiff(cols, [x_name, y_name])

    @assert length(z_names) >= 1

    n_sims = 1000
    n_rows = nrow(data)
    samples = rand(posterior, n_sims) # sample from posterior

    lambda_x_index = sym2range.lambda_x[1][1]
    beta_x_interact_i_index = sym2range.beta_i[1][1]
    beta_x_interact_y_index = sym2range.beta_i[1][2]
    log_variance_intercept_index = sym2range.log_variance_intercept[1][1]
    log_variance_beta_x_interact_i_index = sym2range.log_variance_beta_i[1][1]
    log_variance_beta_x_interact_y_index = sym2range.log_variance_beta_i[1][2]
    intercept_index = sym2range.intercept[1][1]

    beta_x_indices = sym2range.beta_x
    beta_x_indices = map(x -> x[1], beta_x_indices) # convert 27:27 to 27


    x_interact_i = Vector{Float64}(undef, n_rows)
    x_interact_y = Vector{Float64}(undef, n_rows)
    x = @view data[!, x_name]
    z = Matrix(@view data[:, z_names])
    for r in 1:n_rows
        if data.intervention[r] != x_name
            x_interact_i[r] = 0.0
        else 
            x_interact_i[r] = 1.0
        end
        if data.intervention[r] != y_name
            x_interact_y[r] = 0.0
        else 
            x_interact_y[r] = x[r] 
        end
    end

    # rows are data points, columns are sims
    n_quantiles = 5
    yhat = Matrix{Float64}(undef, n_rows, n_sims)
    sigma_hat = Matrix{Float64}(undef, n_rows, n_sims)
    ypred = Matrix{Float64}(undef, n_rows, n_sims)
    ypred_quantiles = Matrix{Float64}(undef, n_rows, n_quantiles)

    for i in 1:n_sims
        yhat[:, i] = samples[intercept_index, i] .+ 
                   hcat(x, z) * samples[beta_x_indices, i] .+
                   samples[beta_x_interact_i_index, i] .* x_interact_i .+
                   samples[beta_x_interact_y_index, i] .* x_interact_y

        sigma_hat[:, i] = exp.(samples[log_variance_intercept_index, i] .+
                          samples[log_variance_beta_x_interact_i_index, i] .* x_interact_i .+
                          samples[log_variance_beta_x_interact_y_index, i] .* x_interact_y)

        dist = MvNormal(yhat[:, i], Diagonal(sqrt.(sigma_hat[:, i])))
        ypred[:, i] = rand(dist, 1)
    end

    for i in 1:n_rows
        ypred_quantiles[i, :] = quantile(yhat[i, :], [0.025, 0.25, 0.50, 0.75, 0.95])
    end
    
    return ypred_quantiles
end

function posterior_predictive(posterior::Vector{Bijectors.MultivariateTransformed}, names::Vector{Vector{String}}, sym2range::Vector{NamedTuple}, data::DataFrame)

    pred = Vector{Matrix{Float64}}(undef, length(posterior))

    @info "$(now()) finding posterior predictive"

    Threads.@threads for i in 1:length(posterior)
    # for i in 1:length(posterior)
        pred[i] = posterior_predictive(posterior[i], names[i], sym2range[i], data)
    end

    return pred
end

function round_numeric_print(df::DataFrame)
    coltypes = eltype.(eachcol(df))

    print_df = deepcopy(df)

    for i in 1:ncol(df)
        if coltypes[i] <: Number
            col = names(df)[i]
            print_df = transform(print_df, Symbol(col) => x -> round.(x, digits = 2), renamecols = false)
        end
    end

    return print_df
end
