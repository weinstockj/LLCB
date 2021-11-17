@model function bayes_linear_lasso(y, x, ::Type{T} = Float64) where {T}
    variance_prior ~ truncated(Normal(0, var(y)), 0, Inf)
    intercept ~ Normal(0.0, 10.0)

    n = size(x, 1)
    p = size(x, 2)
    beta = Vector{T}(undef, p)

    for i in 1:p
       beta[i] ~ Laplace(0.0, 1.0) 
       # beta[i] ~ Normal(0, 1.0)
    end
    # beta ~ MvNormal(p, 1.0)

    yhat = intercept .+ x * beta
    # y ~ MvNormal(yhat, sqrt(variance_prior))
    for i in 1:n
        y[i] ~ Normal(yhat[i], sqrt(variance_prior))
    end
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

function d_connected_advi(x::T, y::T, z::Matrix{Union{Missing, Float64}}) where T<:AbstractArray{Union{Missing, Float64}}
    full_model       = bayes_linear_lasso(y, hcat(x, z))
    # restricted_model = bayes_linear_lasso(y, z) 

    # full_model_chain = sample(full_model, NUTS(0.65), MCMCThreads(), 200, 4)
    advi = ADVI(10, 1000)
    full_model_chain_vi = vi(full_model, advi; optimizer = Flux.ADAM())
    # full_model_chain_vi = vi(full_model, advi)
    # full_model_chain_vi = vi(full_model, advi; optimizer = DecayedADAGrad(1e-2, 1.1, 0.9))
    # @time restricted_model_chain = sample(restricted_model, NUTS(0.65), 1000)

    # return full_model_chain, restricted_model_chain
    return full_model_chain_vi
    # return full_model_chain
end

function d_connected_map(x::T, y::T, z::Matrix{Union{Missing, Float64}}) where T<:AbstractArray{Union{Missing, Float64}}
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

function estimate_skeleton(data::DataFrame)
    cols = setdiff(ko_targets(), ko_controls())
    n_cols = length(cols)
    n_rows = nrow(data)
    n_vals = Int64(n_cols * (n_cols - 1) / 2)

    values  = Array{Bijectors.MultivariateTransformed}(undef, n_vals)
    names   = Array{Array{String}}(undef, n_vals)

    count = 0

    @info "$(now()) Now estimating skeleton"
    # takes 1475 seconds

    Threads.@threads for i in 1:n_cols
    # for i in 1:n_cols
        y_name  = cols[i]
        y = @view data[!, y_name]
        for j in 1:i
            if j == i 
                continue
            end
            x_interact_x = Array{Float64}(undef, n_rows)
            x_interact_y = Array{Float64}(undef, n_rows)
            # x_name  = setdiff(cols, [y_name])[j]
            x_name  = cols[j]
            z_names = setdiff(cols, [x_name, y_name])
            x = @view data[!, x_name]
            z = Matrix(@view data[:, z_names])
            for r in 1:n_rows
                if data.intervention[r] != x_name
                    x_interact_x[r] = 0.0
                else 
                    x_interact_x[r] = x[r] 
                end
                if data.intervention[r] != y_name
                    x_interact_y[r] = 0.0
                else 
                    x_interact_y[r] = x[r] 
                end
            end
            z_interact = hcat(z, hcat(x_interact_x, x_interact_y))
            count = count + 1
            values[count] = d_connected_advi(x, y, z_interact)
            names[count] = [x_name, y_name]
        end
    end

    @info "$(now()) Done estimating skeleton"

    dir = output_dir()

    @info "$(now()) Now serializing"

    serialize(joinpath(dir, "values.store"), values)
    serialize(joinpath(dir, "names.store"), names)

    return values, names
end

function parse_skeleton(posterior::Vector{Bijectors.MultivariateTransformed}, names::Vector{Array{String, N} where N})
    edge_weights = Vector{Float64}
    edge_pips    = Vector{Float64}

    epsilon = 0.01
    for i in 1:length(posterior)
        if isassigned(posterior, i)
            @info "Now working on $(names[i])"
            samples = rand(posterior[i], 1000)
            n_vars = size(samples, 1)
            b1_hat = mean(samples[3])
            b1_pip = mean(abs.(samples[3, :]) .> epsilon)
            push!(edge_weights, b1_hat)
            push!(edge_pips, b1_pip)
        end
    end

    return edge_weights

end
