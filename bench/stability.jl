
# Stability survey: run the refactored pipeline on several independent
# simulated datasets and report per-dataset mixing, calibration and accuracy.
using LinearAlgebra, Statistics, Random, Printf, Dates
using LogDensityProblems, Optim, LineSearches, AdvancedHMC, MCMCDiagnosticTools, DataFrames, Pathfinder

const OUT = open(joinpath(@__DIR__, "stability_output.txt"), "w")
say(args...) = (println(OUT, args...); flush(OUT))
sayf(fmt, args...) = (print(OUT, Printf.format(Printf.Format(fmt), args...)); flush(OUT))

include(joinpath(@__DIR__, "..", "src", "total_effects.jl"))
include(joinpath(@__DIR__, "..", "src", "moment_model.jl"))
include(joinpath(@__DIR__, "..", "src", "fit_llcb.jl"))

function random_cyclic_graph(p::Int; n_edges::Int = 2p, strength = (0.2, 0.6),
                             rho::Float64 = 0.6, rng = Random.GLOBAL_RNG)
    B = zeros(p, p)
    offdiag = [(i, j) for i in 1:p, j in 1:p if i != j]
    for s in randperm(rng, length(offdiag))[1:min(n_edges, length(offdiag))]
        i, j = offdiag[s]
        B[i, j] = (rand(rng, Bool) ? 1 : -1) * (strength[1] + (strength[2] - strength[1]) * rand(rng))
    end
    ρ = maximum(abs.(eigvals(B))); ρ > 0 && (B .*= rho / ρ)
    return B
end

function simulate_experiment(B; n_donors = 3, n_replicates = 10, donor_sd = 0.4,
                             donor_imbalance = 0.6, noise_sd = 0.15, rng = Random.GLOBAL_RNG)
    p = size(B, 1)
    off = donor_sd .* randn(rng, n_donors, p)
    rows = Vector{Vector{Float64}}(); interv = Int[]; donor = Int[]
    for k in vcat(1:p, 0)
        Bk = copy(B); k > 0 && (Bk[:, k] .= 0.0)
        Ak = inv(Matrix{Float64}(I, p, p) - Bk)
        w = fill(1.0 / n_donors, n_donors)
        if k > 0 && donor_imbalance > 0
            w .= (1 - donor_imbalance) / n_donors; w[mod1(k, n_donors)] += donor_imbalance
        end
        counts = max.(1, round.(Int, w .* (n_donors * n_replicates)))
        for d in 1:n_donors, _ in 1:counts[d]
            e = 1.0 .+ view(off, d, :) .+ noise_sd .* randn(rng, p)
            k > 0 && (e[k] = 0.0)
            x = vec(e' * Ak); k > 0 && (x[k] = 0.0)
            push!(rows, x); push!(interv, k); push!(donor, d)
        end
    end
    X = reduce(vcat, transpose.(rows))
    ivm = zeros(Int8, size(X, 1), p)
    for i in axes(X, 1); interv[i] > 0 && (ivm[i, interv[i]] = 1); end
    return X, ivm, donor, interv .== 0
end

function average_precision(score::AbstractMatrix, truth::BitMatrix)
    p = size(truth, 1)
    idx = [(i, j) for i in 1:p, j in 1:p if i != j]
    s = [score[i, j] for (i, j) in idx]; y = [truth[i, j] for (i, j) in idx]
    tp = 0; fp = 0; npos = count(y); ap = 0.0; prev = 0.0
    for k in sortperm(s; rev = true)
        y[k] ? (tp += 1) : (fp += 1)
        ap += (tp / (tp + fp)) * (tp / npos - prev); prev = tp / npos
    end
    return ap
end

function main()
    p = 10; nsim = 6
    say("=== stability survey, ", Dates.now(), " ===")
    sayf("genes = %d, datasets = %d, threads = %d\n", p, nsim, Threads.nthreads())
    sayf("%4s %7s %5s %8s %8s %8s %7s %7s %8s %9s %8s %8s | %8s %8s %8s\n",
         "sim", "prior_s", "weak", "AP_lsfr", "AP_mean", "AP_te", "base", "rhat", "ess_min",
         "div_frac", "cov95", "secs", "cf_AP", "cf_RMSE", "cf_cov")
    rows = DataFrame()
    for sim in 1:nsim
        rng = MersenneTwister(1000 + sim)
        B = random_cyclic_graph(p; n_edges = 2p, rho = 0.6, rng = rng)
        X, ivm, donor, controls = simulate_experiment(B; n_replicates = 10, rng = rng)
        sds = vec(std(view(X, controls, :); dims = 1))
        Btrue = B .* (sds * (1 ./ sds)')
        truth = BitMatrix(abs.(Btrue) .> 1e-12)
        te = estimate_total_effects(X, ivm, donor, controls; verbose = false)
        t0 = time()
        fit, mdl = fit_llcb(te; prior = :studentt, prior_df = 3.0, nchains = 4,
                            n_adapt = 600, n_samples = 600, target_accept = 0.9,
                            seed = 100 + sim, verbose = false)
        secs = time() - t0
        offd = [(i, j) for i in 1:p, j in 1:p if i != j]
        lo = fit.quantiles[0.025]; hi = fit.quantiles[0.975]
        row = (; sim, prior_scale = mdl.prior_scale,
               ap_lsfr = average_precision(1 .- fit.lsfr, truth),
               ap_mean = average_precision(abs.(fit.B_mean), truth),
               ap_te = average_precision(abs.(te.Θ), truth),
               base = count(truth) / length(offd),
               rhat = fit.diagnostics[:rhat_max],
               ess_min = minimum([fit.ess[i, j] for (i, j) in offd]),
               div_frac = fit.diagnostics[:divergence_fraction],
               cov95 = mean([lo[i, j] <= Btrue[i, j] <= hi[i, j] for (i, j) in offd]),
               rmse = sqrt(mean([(fit.B_mean[i, j] - Btrue[i, j])^2 for (i, j) in offd])),
               metric = String(fit.diagnostics[:metric]),
               retried = fit.diagnostics[:retried], seconds = secs)
        cf = nothing
        cf_ok = try
            cf = fit_llcb_closed_form(te; ndraws = 4000, rng = MersenneTwister(sim), verbose = false)
            true
        catch err
            err isa ArgumentError || rethrow()
            false
        end
        if cf_ok
            lo2 = cf.quantiles[0.025]; hi2 = cf.quantiles[0.975]
            row = merge(row, (; weak = length(weak_knockdowns(te)),
                        cf_ap = average_precision(1 .- cf.lsfr, truth),
                        cf_rmse = sqrt(mean([(cf.B_mean[i, j] - Btrue[i, j])^2 for (i, j) in offd])),
                        cf_cov95 = mean([lo2[i, j] <= Btrue[i, j] <= hi2[i, j] for (i, j) in offd])))
        else   # closed form not applicable (a readout lacks an identifiable knockdown)
            row = merge(row, (; weak = length(weak_knockdowns(te)), cf_ap = NaN, cf_rmse = NaN, cf_cov95 = NaN))
        end
        push!(rows, row)
        sayf("%4d %7.3f %5d %8.3f %8.3f %8.3f %7.3f %7.3f %8.0f %9.3f %8.3f %8.1f | %8.3f %8.3f %8.3f\n",
             sim, row.prior_scale, row.weak, row.ap_lsfr, row.ap_mean, row.ap_te, row.base,
             row.rhat, row.ess_min, row.div_frac, row.cov95, secs, row.cf_ap, row.cf_rmse, row.cf_cov95)
    end
    open(joinpath(@__DIR__, "stability_rows.tsv"), "w") do io
        println(io, join(names(rows), "\t"))
        for r in eachrow(rows); println(io, join(collect(r), "\t")); end
    end
    say("")
    sayf("mean  AP(1-lsfr) %.3f | AP(|mean|) %.3f | AP(|total effects|) %.3f | base rate %.3f\n",
         mean(rows.ap_lsfr), mean(rows.ap_mean), mean(rows.ap_te), mean(rows.base))
    sayf("mean  rhat %.3f (max %.3f) | min ESS %.0f | divergences %.2f%% | coverage %.3f | RMSE %.3f\n",
         mean(rows.rhat), maximum(rows.rhat), minimum(rows.ess_min),
         100 * mean(rows.div_frac), mean(rows.cov95), mean(rows.rmse))
    sayf("datasets meeting rhat < 1.05: %d of %d; retried: %d\n",
         count(rows.rhat .< 1.05), nsim, count(rows.retried))
    sayf("closed-form resampling: mean AP %.3f | RMSE %.3f | coverage %.3f\n",
         mean(rows.cf_ap), mean(rows.cf_rmse), mean(rows.cf_cov95))
    say("STABILITY_DONE")
end

try
    main()
catch e
    say("STABILITY_FAILED: ", sprint(showerror, e, catch_backtrace())[1:min(end, 4000)])
end
close(OUT)
