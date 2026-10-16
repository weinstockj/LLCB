
using Dates, Printf, Random, LinearAlgebra, Statistics
const OUT = open(joinpath(@__DIR__, "smoke_output.txt"), "w")
say(a...) = (println(OUT, a...); flush(OUT))
sayf(fmt, a...) = (print(OUT, Printf.format(Printf.Format(fmt), a...)); flush(OUT))
say("julia ", VERSION, ", threads ", Threads.nthreads(), ", BLAS threads ", BLAS.get_num_threads())
t = time()
using InferCausalGraph
using InferCausalGraph: closed_form_coefficients, has_full_total_effects, implied_total_effects
using DataFrames, MCMCChains
sayf("using InferCausalGraph: %.1fs\n", time() - t)
include(joinpath(@__DIR__, "simulate.jl"))

function step(name, f)
    t = time()
    try
        r = f(); sayf("[ok]   %-34s %7.1fs\n", name, time() - t); return r
    catch e
        say("[FAIL] ", name, ": ", sprint(showerror, e, catch_backtrace())[1:min(end, 3500)]); return nothing
    end
end

# 1. closed form, noise-free
step("closed form recovers B exactly", () -> begin
    B = random_cyclic_graph(12; rng = MersenneTwister(3))
    err = maximum(abs.(closed_form_coefficients(implied_total_effects(B)) .- B))
    sayf("       max |B_hat - B| = %.2e\n", err); err < 1e-10 || error("not exact")
end)

# data: the repo's own 5-gene cyclic chain, donor effects applied
rng = MersenneTwister(2024)
B = cyclic_chain(5; weight = 0.5)
df, truth = simulate_experiment(B; n_donors = 3, n_replicates = 10, donor_sd = 0.3,
                                donor_imbalance = 0.5, noise_sd = 0.1, rng = rng)
g = interventionGraph(df)
Bs = truth.B_scaled
offd = [(i, j) for i in 1:5, j in 1:5 if i != j]
rmse(M) = sqrt(mean([(M[i, j] - Bs[i, j])^2 for (i, j) in offd]))
cover(f) = mean([f.quantiles[0.025][i, j] <= Bs[i, j] <= f.quantiles[0.975][i, j] for (i, j) in offd])

te = step("stage 1", () -> estimate_total_effects(g; verbose = false))
sayf("       weak knockdowns: %s; first-stage t: %s\n", string(weak_knockdowns(te)), string(round.(abs.(te.knockdown ./ te.knockdown_se); digits = 1)))
step("closed form on estimates", () -> sayf("       RMSE vs truth %.4f (full total effects: %s)\n",
                                             rmse(closed_form_coefficients(te)), has_full_total_effects(te)))

fit = step("fit_llcb (NUTS, 4 chains)", () -> fit_llcb(te; nchains = 4, n_adapt = 400, n_samples = 400, seed = 7, verbose = false)[1])
fit === nothing || sayf("       RMSE %.4f  cov95 %.3f  rhat_max %.3f  div %.2f%%  metric %s  start %s  BLAS %d  phi sampled %s\n",
    rmse(fit.B_mean), cover(fit), fit.diagnostics[:rhat_max], 100 * fit.diagnostics[:divergence_fraction],
    fit.diagnostics[:metric], fit.diagnostics[:map_start], fit.diagnostics[:blas_threads_during_sampling], fit.diagnostics[:phi] != 1.0)

cf = step("fit_llcb_closed_form (4000 draws)", () -> fit_llcb_closed_form(te; ndraws = 4000, rng = MersenneTwister(1), verbose = false))
cf === nothing || sayf("       RMSE %.4f  cov95 %.3f  singular draws %d\n", rmse(cf.B_mean), cover(cf), cf.diagnostics[:singular_draws])

leg = step("fit_cyclic_model_legacy (ported)", () -> fit_cyclic_model_legacy(
    g, false, get_model_params(false, 1.0, 0.01), get_sampling_params(false); rng = MersenneTwister(11)))
if leg !== nothing
    edges = get_cyclic_matrices(g, false)[3]
    parsed = step("parse_cyclic_chain", () -> parse_cyclic_chain(leg[1], leg[2], edges; targets = ["gene_$i" for i in 1:5]))
    if parsed !== nothing
        est = zeros(5, 5)
        for r in eachrow(parsed)
            est[parse(Int, split(r.row, "_")[2]), parse(Int, split(r.col, "_")[2])] = r.estimate
        end
        sayf("       legacy RMSE %.4f; rows %d; columns %s\n", rmse(est), nrow(parsed), join(names(parsed), ","))
        truth_edges = Set((i, j) for (i, j) in offd if Bs[i, j] != 0)
        det = Set((parse(Int, split(r.row, "_")[2]), parse(Int, split(r.col, "_")[2])) for r in eachrow(parsed) if r.PIP > 0.95)
        sayf("       legacy PIP>0.95: %d detected, %d of %d true edges\n", length(det), length(intersect(det, truth_edges)), length(truth_edges))
    end
end
say("SMOKE_DONE")
close(OUT)
