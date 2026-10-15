
# End-to-end run of the refactored pipeline on simulated data:
# stage 1 -> moment model -> MAP -> NUTS -> edge table, scored against truth.
using LinearAlgebra, Statistics, Random, Printf, Dates
using LogDensityProblems, Optim, AdvancedHMC, MCMCDiagnosticTools, DataFrames, Pathfinder
using Pathfinder.Optim.LineSearches

const OUT = open(joinpath(@__DIR__, "fit_demo_output.txt"), "w")
say(args...) = (println(OUT, args...); flush(OUT))
sayf(fmt, args...) = (print(OUT, Printf.format(Printf.Format(fmt), args...)); flush(OUT))

include(joinpath(@__DIR__, "..", "src", "total_effects.jl"))
include(joinpath(@__DIR__, "..", "src", "moment_model.jl"))
include(joinpath(@__DIR__, "..", "src", "fit_llcb.jl"))

# ---- simulation (same generator as validate_core.jl) ----------------------
function random_cyclic_graph(p::Int; n_edges::Int = 2p, strength = (0.2, 0.6),
                             rho::Float64 = 0.6, rng = Random.GLOBAL_RNG)
    B = zeros(p, p)
    offdiag = [(i, j) for i in 1:p, j in 1:p if i != j]
    sel = randperm(rng, length(offdiag))[1:min(n_edges, length(offdiag))]
    for s in sel
        i, j = offdiag[s]
        mag = strength[1] + (strength[2] - strength[1]) * rand(rng)
        B[i, j] = rand(rng, Bool) ? mag : -mag
    end
    ρ = maximum(abs.(eigvals(B))); ρ > 0 && (B .*= rho / ρ)
    return B
end

function simulate_experiment(B; n_donors = 3, n_replicates = 10, donor_sd = 0.3,
                             donor_imbalance = 0.0, noise_sd = 0.1, baseline = 1.0,
                             rng = Random.GLOBAL_RNG)
    p = size(B, 1)
    donor_offset = donor_sd .* randn(rng, n_donors, p)
    rows = Vector{Vector{Float64}}(); interv = Int[]; donor = Int[]
    for k in vcat(1:p, 0)
        Bk = copy(B); k > 0 && (Bk[:, k] .= 0.0)
        Ak = inv(Matrix{Float64}(I, p, p) - Bk)
        w = fill(1.0 / n_donors, n_donors)
        if k > 0 && donor_imbalance > 0
            focus = mod1(k, n_donors)
            w .= (1 - donor_imbalance) / n_donors; w[focus] += donor_imbalance
        end
        counts = max.(1, round.(Int, w .* (n_donors * n_replicates)))
        for d in 1:n_donors, _ in 1:counts[d]
            e = baseline .+ view(donor_offset, d, :) .+ noise_sd .* randn(rng, p)
            k > 0 && (e[k] = 0.0)
            x = vec(e' * Ak); k > 0 && (x[k] = 0.0)
            push!(rows, x); push!(interv, k); push!(donor, d)
        end
    end
    X = reduce(vcat, transpose.(rows))
    ivm = zeros(Int8, size(X, 1), size(X, 2))
    for i in axes(X, 1); interv[i] > 0 && (ivm[i, interv[i]] = 1); end
    return X, ivm, donor, interv .== 0
end

# ---- precision / recall on edge presence ---------------------------------
function pr_curve(score::Matrix{Float64}, truth::BitMatrix)
    p = size(truth, 1)
    idx = [(i, j) for i in 1:p, j in 1:p if i != j]
    s = [score[i, j] for (i, j) in idx]
    y = [truth[i, j] for (i, j) in idx]
    ord = sortperm(s; rev = true)
    tp = 0; fp = 0; npos = count(y)
    ap = 0.0; prev_recall = 0.0
    for k in ord
        y[k] ? (tp += 1) : (fp += 1)
        prec = tp / (tp + fp); rec = tp / npos
        ap += prec * (rec - prev_recall); prev_recall = rec
    end
    return ap, npos, length(idx)
end

function main()
    say("=== end-to-end fit, ", Dates.now(), " ===")
    say("threads = ", Threads.nthreads())
    p = 8
    rng = MersenneTwister(4242)
    B = random_cyclic_graph(p; n_edges = 2p, rho = 0.6, rng = rng)
    X, ivm, donor, controls = simulate_experiment(
        B; n_donors = 3, n_replicates = 10, donor_sd = 0.4,
        donor_imbalance = 0.6, noise_sd = 0.15, rng = rng)
    sds = vec(std(view(X, controls, :); dims = 1))
    Btrue = B .* (sds * (1 ./ sds)')          # scale estimate_total_effects targets
    truth = BitMatrix(abs.(Btrue) .> 1e-12)
    sayf("simulated %d samples, %d genes, %d true edges\n", size(X, 1), p, count(truth))

    t = time()
    te = estimate_total_effects(X, ivm, donor, controls; verbose = false)
    sayf("stage 1: %.2fs\n", time() - t)

    fit, mdl = fit_llcb(te; prior = :studentt, prior_scale = 0.1, prior_df = 3.0,
                        nchains = 4, n_adapt = 400, n_samples = 400,
                        use_pathfinder = true, seed = 7, verbose = false)
    d = fit.diagnostics
    sayf("MAP   : %.2fs  logp = %.2f\n", d[:map_seconds], d[:map_logp])
    sayf("NUTS  : %.1fs for %d chains x %d draws (dim = %d)\n",
         d[:sample_seconds], d[:nchains], d[:n_samples], llcb_dimension(mdl))
    sayf("        step sizes %s\n", string(round.(d[:step_size]; digits = 4)))
    sayf("        divergences %s, mean accept %.3f\n",
         string(d[:divergences]), mean(d[:acceptance]))
    sayf("        dispersion phi = %.3f\n", d[:phi])
    sayf("        spectral radius: MAP %.3f, posterior mean %.3f (true %.3f)\n",
         d[:spectral_radius_map], d[:spectral_radius_posterior_mean],
         maximum(abs.(eigvals(Btrue))))

    offd = [(i, j) for i in 1:p, j in 1:p if i != j]
    rhats = [fit.rhat[i, j] for (i, j) in offd]
    esss  = [fit.ess[i, j] for (i, j) in offd]
    sayf("        rhat  max %.3f, median %.3f\n", maximum(rhats), median(rhats))
    sayf("        ess   min %.0f, median %.0f\n", minimum(esss), median(esss))

    ap_lsfr, npos, ntot = pr_curve(1 .- fit.lsfr, truth)
    ap_map, _, _ = pr_curve(abs.(fit.map_B), truth)
    ap_mean, _, _ = pr_curve(abs.(fit.B_mean), truth)
    ap_te, _, _ = pr_curve(abs.(vcat(te.Θ)), truth)
    sayf("average precision: 1-lsfr %.3f | |posterior mean| %.3f | |MAP| %.3f | |total effects| %.3f (base rate %.3f)\n",
         ap_lsfr, ap_mean, ap_map, ap_te, npos / ntot)

    err = [fit.B_mean[i, j] - Btrue[i, j] for (i, j) in offd]
    sayf("coefficient RMSE (posterior mean) %.4f\n", sqrt(mean(err .^ 2)))
    lo = fit.quantiles[0.025]; hi = fit.quantiles[0.975]
    cov = mean([lo[i, j] <= Btrue[i, j] <= hi[i, j] for (i, j) in offd])
    sayf("95%% credible interval coverage of true B: %.3f\n", cov)

    tbl = parse_llcb_fit(fit; targets = ["gene_$i" for i in 1:p])
    sort!(tbl, :lsfr)
    say("top 10 edges by lsfr:")
    for r in eachrow(first(tbl, 10))
        sayf("  %-8s -> %-8s est %+.3f [%+.3f, %+.3f] lsfr %.3f PIP %.2f rhat %.3f true %+.3f\n",
             r.row, r.col, r.estimate, r[Symbol("2.5%")], r[Symbol("97.5%")],
             r.lsfr, r.PIP, r.rhat,
             Btrue[parse(Int, split(r.row, "_")[2]), parse(Int, split(r.col, "_")[2])])
    end
    say("FIT_DEMO_OK")
end

try
    main()
catch e
    say("FIT_DEMO_FAILED: ", sprint(showerror, e, catch_backtrace())[1:min(end, 4000)])
end
close(OUT)
