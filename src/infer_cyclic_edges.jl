"""
`fit_cyclic_model(g, false, model_pars, sampling_pars)`

This fits the LLCB model itself. The  first argument must
be an object of class `interventionGraph` . 

```julia-repl
    model = fit_cyclic_model(graph, false, model_pars, sampling_pars)
```

Returns:
1. The MCMCChains object from pathfinder
2. The posterior adjacency matrix samples
3. The Turing model
4. The entire pathfinder object

"""
function fit_cyclic_model(g::interventionGraph, log_normalize::Bool, model_pars::NamedTuple, sampling_pars::NamedTuple)
    Turing.setadbackend(:forwarddiff)

    @info "$(now()) preparing matrices for the model"
    T, t, edges = get_cyclic_matrices(g, log_normalize, true, true)
    T = sparse(T)

    model = joint_cyclic_model(g.nv, model_pars, T, t)

    init_β = 0.3 .* (T \ t)
    noise_dist = Normal(0, .10)
    init = vcat(
        # zeros(g.nv * (g.nv - 1)), #beta
        init_β,
        # -5.0
        fill(-3.0, g.nv) # σ
    )
    @info "$(now()) running pathfinder now"
    nruns = 5
    # 0.72 seconds with gradientdescent()
    # 0.10 with LBFGS
    # result_multi = multipathfinder(
    result_multi = wrap_multipathfinder(
        model, 
        2_000;
        nruns = nruns,
        # ad_backend = AD.ReverseDiffBackend(),
        optimizer = Optim.LBFGS(; 
            m = 6,
            linesearch = HagerZhang(), 
            alphaguess = InitialHagerZhang(α0=0.8),
        ),
        init = [Float64.(init .+ rand(noise_dist, length(init))) for i in 1:nruns],
        # init_scale = 0.01,
        importance = true,
        ndraws_elbo = 100,
        # show_trace = false,
        # ntries = 100
        # iterations = 200
    )
    @info "$(now()) done running pathfinder"

    # @info "$(now()) running MCMC now"
    # sampler = NUTS(
    #         # sampling_pars[:warmups],
    #         300,
    #         sampling_pars[:acceptance];
    #         max_depth = sampling_pars[:max_depth],
    #         init_ϵ = sampling_pars[:init_ϵ]
    # )

    # model_chain = sample(
    #         model, 
    #         sampler,
    #         MCMCThreads(), 
    #         # MCMCSerial(), 
    #         # sampling_pars[:samples], 
    #         300,
    #         Threads.nthreads();
    #         # 2;
    #         init_params = init
    #     
    # )

    model_chain = result_multi.draws_transformed 
    chains_params = Turing.MCMCChains.get_sections(model_chain, :parameters)
    quantities = generated_quantities(model, chains_params)
    # return model_chain, quantities, path_init, sampler, vec_β_init
    return model_chain, quantities, model, result_multi
end

"""
`parse_cyclic_chain(chains, posterior_adjacency, edges)`

```julia-repl
    model = fit_cyclic_model(graph, false, model_pars, sampling_pars)
    parsed = parse_cyclic_chain(
        model[1], model[2], cyclic_matrices[3]
    )
```

"""
function parse_cyclic_chain(chains::Chains, posterior_adjacency::Matrix{Matrix{Float64}}, edges::Vector{Pair}; targets=setdiff(ko_targets(), ko_controls()))

    n_samples = size(posterior_adjacency, 1) # slightly misleading - total mcmc samples is n_samples * n_chains
    n_chains  = size(posterior_adjacency, 2)

    @assert n_samples >= 10
    @assert n_chains  >= 1

    posterior_mean = mean(mean(posterior_adjacency; dims = 2)) # inner mean takes mean across chains, outer averages across samples
    ϵ = .05
    nv = size(posterior_mean, 1)
    @assert nv == size(posterior_mean, 2)

    posterior_pip = mean(
        abs.(
            reduce(hcat, vec.(vec(posterior_adjacency)))
        ) .> ϵ;
        dims = 2
    )
    posterior_pip = reshape(posterior_pip, nv, nv)

    posterior_lsfr_pos = mean(
        (
            reduce(hcat, vec.(vec(posterior_adjacency)))
        ) .>= 0;
        dims = 2
    )
    posterior_lsfr_neg = mean(
        (
            reduce(hcat, vec.(vec(posterior_adjacency)))
        ) .< 0;
        dims = 2
    )

    posterior_lsfr = min.(posterior_lsfr_pos, posterior_lsfr_neg)
    posterior_lsfr = reshape(posterior_lsfr, nv, nv)

    result = DataFrame() # convert to 'long' format

    for (z, e) in enumerate(edges)
        i = e.first
        j = e.second
        row_label = targets[i]
        col_label = targets[j]

            if i == j

                row = (;
                    :row => row_label,
                    :col => col_label,
                    :estimate => 0.0,
                    Symbol("2.5%") => 0.0,
                    Symbol("97.5%") => 0.0,
                    :PIP => 0.0,
                    :lsfr => 1.0,
                    :rhat => 1.0,
                    :ess => Float64(n_samples * n_chains),
                    :std => 0.0
                )

            else
                # (1, 2) -> (1)
                # 2 - Int(2 > 1) == 1
                # (3, 1) -> (2) * (nv - 1) + 1 - Int(1 > 3) = 2 * nv - 2 + 1 - 0
                # alt_idx = (i - 1) * (nv - 1) + j - Int(j > i)
                alt_idx = z
                summary   = summarize(chains[[Symbol("β[$alt_idx]")]])
                quantiles = quantile(chains[[Symbol("β[$alt_idx]")]])

                row = (;
                    :row => row_label,
                    :col => col_label,
                    :estimate => summary[:, :mean],
                    Symbol("2.5%") => quantiles[:, Symbol("2.5%")],
                    Symbol("97.5%") => quantiles[:, Symbol("97.5%")],
                    :PIP => posterior_pip[j, i],
                    :lsfr => posterior_lsfr[j, i],
                    :rhat => summary[:, :rhat],
                    :ess => summary[:, :ess_bulk],
                    :std => summary[:, :std]
                )

            end

            push!(result, row)
        # end
    end

    return result
end


# function Pathfinder.multipathfinder(
function wrap_multipathfinder(
        model::DynamicPPL.Model,
        ndraws::Int,
        ad_backend = AD.ReverseDiffBackend();
        rng=Random.GLOBAL_RNG,
        use_elbo = true,
        optimizer,
        ndraws_elbo = 50,
        init,
        init_scale=0.03,
        init_sampler=Pathfinder.UniformSampler(init_scale),
        # init = UniformSampler(0.1),
        nruns::Int,
        ntries = 1500,
        importance = false,
        kwargs...
    )
    var_names = Pathfinder.flattened_varnames_list(model)
    fun = Turing.optim_function(model, Turing.MAP(); constrained=false)
    init1 = fun.init()
    init2 = [init_sampler(rng, init1)]
    for _ in 2:nruns
        push!(init2, init_sampler(rng, deepcopy(init1)))
    end

    dim = length(init1)
    result = multipathfinder(
    # # result = pathfinder(
        fun.func, ndraws;
        # logp, 
        # ∇logp,
        # ndraws; 
    #     # dim = dim,
    #     # optimizer = optimizer,
    #   
        rng, 
        optimizer = Optim.LBFGS(;m=8),
        ndraws_elbo = ndraws_elbo,
        importance = importance,
        nruns = nruns,
        input = model, 
        init = init2, 
        ntries = ntries,
        kwargs...
    )
    local draws
    if use_elbo
        draws = reduce(vcat, transpose.(fun.transform.(eachcol(result.draws))))
    else
        draws = rand(last(result.pathfinder_results[1].fit_distributions), ndraws)
        draws = reduce(
            vcat, 
            transpose.(fun.transform.(eachcol(draws)))
        )
    end
    chns = MCMCChains.Chains(draws, var_names; info=(; pathfinder_result=result))
    result_new = Accessors.@set result.draws_transformed = chns
    return result_new
end
