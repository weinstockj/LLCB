
# =============================================================================
# Benchmark: accuracy, calibration and runtime of the refactored pipeline,
# with the two changes separated.
#
#   arm "total_effects"   : rank edges by |stage-1 total effect| (no stage 2)
#   arm "legacy1_new2"    : published pooled-OLS stage 1, new stage 2
#   arm "new1_new2"       : donor-adjusted stage 1, new stage 2
#
# Comparing the last two isolates the contribution of stage 1 inside an
# identical inference engine; comparing against the first shows what stage 2
# buys over reading the total effects directly.
# =============================================================================
using LinearAlgebra, Statistics, Random, Printf, Dates
using LogDensityProblems, Optim, AdvancedHMC, MCMCDiagnosticTools, DataFrames, Pathfinder, CSV
using Pathfinder.Optim.LineSearches

const OUT = open(joinpath(@__DIR__, "benchmark_output.txt"), "w")
say(args...) = (println(OUT, args...); flush(OUT))
sayf(fmt, args...) = (print(OUT, Printf.format(Printf.Format(fmt), args...)); flush(OUT))

include(joinpath(@__DIR__, "..", "src", "total_effects.jl"))
include(joinpath(@__DIR__, "..", "src", "moment_model.jl"))
include(joinpath(@__DIR__, "..", "src", "fit_llcb.jl"))

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

function simulate_experiment(B; n_donors = 3, n_replicates = 10, donor_sd = 0.4,
                             donor_imbalance = 0.6, noise_sd = 0.15, baseline = 1.0,
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

function legacy_total_effects(X, ivm, controls)
    p = size(X, 2)
    x = X .- mean(view(X, controls, :); dims = 1)
    sds = vec(std(view(X, controls, :); dims = 1)); sds[sds .<= 0] .= 1.0
    x = x ./ sds'
    T = zeros(p, p)
    for i in 1:p
        obs = findall((view(ivm, :, i) .== 1) .| collect(controls))
        A = hcat(ones(length(obs)), x[obs, i])
        for j in 1:p
            i == j && continue
            T[i, j] = (A \ x[obs, j])[2]
        end
    end
    return T
end

# average precision for ranking off-diagonal edges
function average_precision(score::AbstractMatrix, truth::BitMatrix)
    p = size(truth, 1)
    idx = [(i, j) for i in 1:p, j in 1:p if i != j]
    s = [score[i, j] for (i, j) in idx]; y = [truth[i, j] for (i, j) in idx]
    ord = sortperm(s; rev = true)
    tp = 0; fp = 0; npos = count(y); ap = 0.0; prev = 0.0
    for k in ord
        y[k] ? (tp += 1) : (fp += 1)
        ap += (tp / (tp + fp)) * (tp / npos - prev); prev = tp / npos
    end
    return ap
end

# wrap a plain total-effect matrix as a TotalEffectEstimate so the legacy
# stage-1 output can be fed to the same stage 2 (its standard errors are not
# available, so a common scale is used -- which is exactly the assumption the
# published stage 2 makes)
function as_estimate(Θ::Matrix{Float64}, se::Float64, targets)
    p = size(Θ, 1)
    Θ2 = copy(Θ); for u in 1:p; Θ2[u, u] = 0.0; end
    active = trues(p, p); for u in 1:p; active[u, u] = false; end
    return TotalEffectEstimate(Θ2, fill(se, p, p), nothing, fill(-1.0, p), active,
                               fill(20, p), fill(15, p), collect(1:p), targets)
end

function main()
    p = 10
    nsim = 5
    targets = ["gene_$i" for i in 1:p]
    say("=== LLCB benchmark, ", Dates.now(), " ===")
    sayf("genes = %d, simulations = %d, threads = %d\n", p, nsim, Threads.nthreads())
    say("arms: total_effects (stage 1 only) | legacy1_new2 | new1_new2")
    rows = DataFrame()

    for sim in 1:nsim
        rng = MersenneTwister(1000 + sim)
        B = random_cyclic_graph(p; n_edges = 2p, rho = 0.6, rng = rng)
        X, ivm, donor, controls = simulate_experiment(B; n_replicates = 10, rng = rng)
        sds = vec(std(view(X, controls, :); dims = 1))
        Btrue = B .* (sds * (1 ./ sds)')
        truth = BitMatrix(abs.(Btrue) .> 1e-12)

        t = time(); te_new = estimate_total_effects(X, ivm, donor, controls; verbose = false)
        t_s1_new = time() - t
        t = time(); Θ_old = legacy_total_effects(X, ivm, controls); t_s1_old = time() - t
        # give the legacy arm the median new standard error so the two stage-2
        # runs differ only in the point estimates and their weighting
        se_common = median(filter(isfinite, vec(te_new.se)))
        te_old = as_estimate(Θ_old, se_common, targets)

        push!(rows, (; sim, arm = "total_effects", ap = average_precision(abs.(te_new.Θ), truth),
                     ap_legacy_te = average_precision(abs.(Θ_old), truth),
                     rmse = NaN, coverage = NaN, rhat_max = NaN, ess_min = NaN,
                     divergences = 0, seconds = t_s1_new))

        for (arm, te) in (("legacy1_new2", te_old), ("new1_new2", te_new))
            t0 = time()
            fit, mdl = fit_llcb(te; prior = :studentt, prior_scale = 0.1, prior_df = 3.0,
                                nchains = 4, n_adapt = 600, n_samples = 600,
                                target_accept = 0.9, use_pathfinder = true,
                                seed = 100 + sim, verbose = false)
            secs = time() - t0
            offd = [(i, j) for i in 1:p, j in 1:p if i != j]
            lo = fit.quantiles[0.025]; hi = fit.quantiles[0.975]
            push!(rows, (; sim, arm,
                         ap = average_precision(1 .- fit.lsfr, truth),
                         ap_legacy_te = NaN,
                         rmse = sqrt(mean([(fit.B_mean[i, j] - Btrue[i, j])^2 for (i, j) in offd])),
                         coverage = mean([lo[i, j] <= Btrue[i, j] <= hi[i, j] for (i, j) in offd]),
                         rhat_max = maximum([fit.rhat[i, j] for (i, j) in offd]),
                         ess_min = minimum([fit.ess[i, j] for (i, j) in offd]),
                         divergences = sum(fit.diagnostics[:divergences]),
                         seconds = secs))
            sayf("  sim %d %-14s ap=%.3f rmse=%.3f cov=%.3f rhat=%.3f ess=%.0f div=%d %.1fs\n",
                 sim, arm, last(rows).ap, last(rows).rmse, last(rows).coverage,
                 last(rows).rhat_max, last(rows).ess_min, last(rows).divergences, secs)
        end
    end

    CSV.write(joinpath(@__DIR__, "benchmark_rows.csv"), rows)
    say("")
    say("=== summary (mean over simulations) ===")
    sayf("%-16s %8s %8s %9s %9s %9s %7s %9s\n",
         "arm", "AP", "RMSE", "coverage", "rhat max", "ess min", "div", "seconds")
    for arm in ("total_effects", "legacy1_new2", "new1_new2")
        sub = rows[rows.arm .== arm, :]
        f(c) = mean(filter(!isnan, sub[!, c]))
        sayf("%-16s %8.3f %8.3f %9.3f %9.3f %9.0f %7.1f %9.1f\n",
             arm, f(:ap), f(:rmse), f(:coverage), f(:rhat_max), f(:ess_min),
             mean(sub.divergences), f(:seconds))
    end
    sub = rows[rows.arm .== "total_effects", :]
    sayf("%-16s %8.3f  (ranking by |legacy pooled-OLS total effect|)\n",
         "legacy_te", mean(filter(!isnan, sub.ap_legacy_te)))
    say("")
    say("BENCHMARK_DONE")
end

try
    main()
catch e
    say("BENCHMARK_FAILED: ", sprint(showerror, e, catch_backtrace())[1:min(end, 4000)])
end
close(OUT)
