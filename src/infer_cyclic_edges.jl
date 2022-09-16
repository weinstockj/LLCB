function fit_cyclic_model(g::interventionGraph, log_normalize::Bool, model_pars::NamedTuple, sampling_pars::NamedTuple)
    Turing.setadbackend(:forwarddiff)

    μ = mean(normalize(g.x) .* (1.0 .- g.interventions))

    T, t, edges = get_cyclic_matrices(g, log_normalize)

    model = joint_cyclic_model(g.nv, model_pars, T, t)

    init = vcat(
        zeros(g.nv * (g.nv - 1)), #beta
        -1.0
    )
    @info "$(now()) running MCMC now: debug 2"
    sampler = NUTS(
            # sampling_pars[:warmups],
            300,
            sampling_pars[:acceptance];
            max_depth = sampling_pars[:max_depth],
            init_ϵ = sampling_pars[:init_ϵ]
    )

    model_chain = sample(
            model, 
            sampler,
            MCMCThreads(), 
            # MCMCSerial(), 
            # sampling_pars[:samples], 
            300,
            Threads.nthreads();
            # 2;
            init_params = init
        
    )

    chains_params = Turing.MCMCChains.get_sections(model_chain, :parameters)
    quantities = generated_quantities(model, chains_params)
    # return model_chain, quantities, path_init, sampler, vec_β_init
    return model_chain, quantities
end

function parse_cyclic_chain(chains::Chains, posterior_adjacency::Matrix{Matrix{Float64}}, edges::Vector{Pair}; targets=setdiff(ko_targets(), ko_controls()))

    n_samples = size(posterior_adjacency, 1) # slightly misleading - total mcmc samples is n_samples * n_chains
    n_chains  = size(posterior_adjacency, 2)

    @assert n_samples >= 50
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
                    :ess => summary[:, :ess],
                    :std => summary[:, :std]
                )

            end

            push!(result, row)
        # end
    end

    return result
end

