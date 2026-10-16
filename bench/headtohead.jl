
# Head-to-head on the v0.3.0 stack: ported published stage 2 vs fit_llcb vs closed form.
using Dates, Printf, Random, LinearAlgebra, Statistics
const OUT = open(joinpath(@__DIR__, "headtohead_output.txt"), "w")
say(a...) = (println(OUT, a...); flush(OUT))
sayf(fmt, a...) = (print(OUT, Printf.format(Printf.Format(fmt), a...)); flush(OUT))
using InferCausalGraph, DataFrames, MCMCChains
include(joinpath(@__DIR__, "simulate.jl"))

function average_precision(score, truth)
    p = size(truth, 1); idx = [(i, j) for i in 1:p, j in 1:p if i != j]
    s = [score[i, j] for (i, j) in idx]; y = [truth[i, j] for (i, j) in idx]
    tp = 0; fp = 0; npos = count(y); ap = 0.0; prev = 0.0
    for k in sortperm(s; rev = true)
        y[k] ? (tp += 1) : (fp += 1); ap += (tp / (tp + fp)) * (tp / npos - prev); prev = tp / npos
    end
    ap
end

p = 8; nsim = 6
say("=== head-to-head, ", Dates.now(), ", julia ", VERSION, ", threads ", Threads.nthreads(), " ===")
sayf("%4s %-9s %8s %8s %8s %8s %8s %8s\n", "sim", "arm", "AP", "RMSE", "cov95", "detect", "rhat", "secs")
rows = DataFrame()
offd = [(i, j) for i in 1:p, j in 1:p if i != j]
for sim in 1:nsim
    rng = MersenneTwister(5000 + sim)
    B = random_cyclic_graph(p; n_edges = 2p, rho = 0.6, rng = rng)
    df, truth = simulate_experiment(B; n_donors = 3, n_replicates = 10, donor_sd = 0.4,
                                    donor_imbalance = 0.6, noise_sd = 0.15, rng = rng)
    g = interventionGraph(df); Bs = truth.B_scaled
    T = BitMatrix(abs.(Bs) .> 1e-12)
    rec(arm, score, est, lo, hi, det, rh, secs) = begin
        r = (; sim, arm, ap = average_precision(score, T),
             rmse = sqrt(mean([(est[i, j] - Bs[i, j])^2 for (i, j) in offd])),
             cov95 = mean([lo[i, j] <= Bs[i, j] <= hi[i, j] for (i, j) in offd]),
             tp = count(det .& T), fp = count(det .& .!T), rhat = rh, secs)
        push!(rows, r)
        sayf("%4d %-9s %8.3f %8.3f %8.3f %4d/%-3d %8.3f %8.1f\n", sim, arm, r.ap, r.rmse, r.cov95, r.tp, r.fp, rh, secs)
    end
    # published method (ported), published defaults
    try
        t0 = time()
        leg = fit_cyclic_model_legacy(g, false, get_model_params(false, 1.0, 0.01), get_sampling_params(false);
                                      rng = MersenneTwister(sim))
        parsed = parse_cyclic_chain(leg[1], leg[2], get_cyclic_matrices(g, false)[3];
                                    targets = ["gene_$i" for i in 1:p])
        secs = time() - t0
        est = zeros(p, p); pip = zeros(p, p); lo = zeros(p, p); hi = zeros(p, p); rh = 0.0
        for r in eachrow(parsed)
            i = parse(Int, split(r.row, "_")[2]); j = parse(Int, split(r.col, "_")[2])
            est[i, j] = r.estimate; pip[i, j] = r.PIP; lo[i, j] = r[Symbol("2.5%")]; hi[i, j] = r[Symbol("97.5%")]
            rh = max(rh, r.rhat)
        end
        rec("published", pip, est, lo, hi, pip .> 0.95, rh, secs)
    catch e
        say("  sim $sim published FAILED: ", sprint(showerror, e)[1:min(end, 600)])
    end
    te = estimate_total_effects(g; verbose = false)
    t0 = time()
    fit = fit_llcb(te; nchains = 4, n_adapt = 600, n_samples = 600, target_accept = 0.9, seed = sim, verbose = false)[1]
    secs = time() - t0
    rec("fit_llcb", 1 .- fit.lsfr, fit.B_mean, fit.quantiles[0.025], fit.quantiles[0.975],
        fit.lsfr .< 0.05, fit.diagnostics[:rhat_max], secs)
    if has_full_total_effects(te)
        t0 = time(); cf = fit_llcb_closed_form(te; ndraws = 4000, rng = MersenneTwister(sim), verbose = false)
        rec("closed", 1 .- cf.lsfr, cf.B_mean, cf.quantiles[0.025], cf.quantiles[0.975], cf.lsfr .< 0.05, NaN, time() - t0)
    end
    sayf("     weak knockdowns: %d\n", length(weak_knockdowns(te)))
end
say("--- means by arm ---")
for gdf in groupby(rows, :arm)
    sayf("%-9s n=%d AP %.3f RMSE %.3f cov95 %.3f TP %.1f FP %.1f secs %.1f\n", gdf.arm[1], nrow(gdf),
         mean(gdf.ap), mean(gdf.rmse), mean(gdf.cov95), mean(gdf.tp), mean(gdf.fp), mean(gdf.secs))
end
open(joinpath(@__DIR__, "headtohead_rows.tsv"), "w") do io
    println(io, join(names(rows), '\t')); for r in eachrow(rows); println(io, join(collect(r), '\t')); end
end
say("H2H_DONE"); close(OUT)
