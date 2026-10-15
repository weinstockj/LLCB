
# =============================================================================
# Validation of the LLCB numerical core, using nothing outside the standard
# library: the stage-1 estimator, the exact total-effect identity, and the
# analytic gradient.
# =============================================================================
using LinearAlgebra, Statistics, Random, Printf

const OUT = open(joinpath(@__DIR__, "validate_core_output.txt"), "w")
say(args...) = (println(OUT, args...); flush(OUT))
sayf(fmt, args...) = (print(OUT, Printf.format(Printf.Format(fmt), args...)); flush(OUT))

include(joinpath(@__DIR__, "..", "src", "total_effects.jl"))
include(joinpath(@__DIR__, "..", "src", "moment_model.jl"))

const RESULTS = Dict{String,Any}()
pass(name, ok) = (RESULTS[name] = ok; sayf("  %-44s %s\n", name, ok ? "PASS" : "FAIL"); ok)

# ---------------------------------------------------------------------------
# simulation of a cyclic system under hard interventions
# ---------------------------------------------------------------------------
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
    ρ = maximum(abs.(eigvals(B)))
    ρ > 0 && (B .*= rho / ρ)
    return B
end

function simulate_experiment(B; n_donors = 3, n_replicates = 10, donor_sd = 0.3,
                             donor_imbalance = 0.0, noise_sd = 0.1, baseline = 1.0,
                             rng = Random.GLOBAL_RNG)
    p = size(B, 1)
    donor_offset = donor_sd .* randn(rng, n_donors, p)
    rows = Vector{Vector{Float64}}()
    interv = Int[]; donor = Int[]
    for k in vcat(1:p, 0)
        Bk = copy(B); k > 0 && (Bk[:, k] .= 0.0)
        Ak = inv(Matrix{Float64}(I, p, p) - Bk)
        w = fill(1.0 / n_donors, n_donors)
        if k > 0 && donor_imbalance > 0
            focus = mod1(k, n_donors)
            w .= (1 - donor_imbalance) / n_donors
            w[focus] += donor_imbalance
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
    n = size(X, 1); p2 = size(X, 2)
    ivm = zeros(Int8, n, p2)
    for i in 1:n
        interv[i] > 0 && (ivm[i, interv[i]] = 1)
    end
    return X, ivm, donor, interv .== 0
end

# ---------------------------------------------------------------------------
# 1. the exact total-effect identity
# ---------------------------------------------------------------------------
say("=== 1. hard-intervention total-effect identity ===")
let
    rng = MersenneTwister(11)
    worst_rec = 0.0; worst_sim = 0.0; worst_dag = 0.0
    for trial in 1:20
        p = 7
        B = random_cyclic_graph(p; rng = rng)
        T = implied_total_effects(B)
        # (a) satisfies the recursion the original code encoded as T beta = t
        for u in 1:p, v in 1:p
            u == v && continue
            rhs = sum(T[u, w] * B[w, v] for w in 1:p if w != v)
            worst_rec = max(worst_rec, abs(T[u, v] - rhs))
        end
        # (b) matches the brute-force per-node computation B[u,-u] (I - B[-u,-u])^-1
        for u in 1:p
            keep = setdiff(1:p, u)
            direct = inv(Matrix{Float64}(I, p - 1, p - 1) - B[keep, keep])
            brute = vec(B[u, keep]' * direct)
            worst_sim = max(worst_sim, maximum(abs.(brute .- T[u, keep])))
        end
        # (c) equals (I - B)^-1 when the graph is acyclic, and not otherwise
        Bdag = triu(abs.(B), 1) .* 0.3
        Adag = inv(Matrix{Float64}(I, p, p) - Bdag)
        worst_dag = max(worst_dag, maximum(abs.(implied_total_effects(Bdag) .- Adag)))
    end
    sayf("  max residual of the parent recursion        %.3e\n", worst_rec)
    sayf("  max deviation from per-node brute force     %.3e\n", worst_sim)
    sayf("  max deviation from (I-B)^-1 on a DAG        %.3e\n", worst_dag)
    pass("moment identity", worst_rec < 1e-10)
    pass("agrees with per-node hard-intervention solve", worst_sim < 1e-10)
    pass("reduces to (I-B)^-1 on a DAG", worst_dag < 1e-10)
    # and is genuinely different on a cyclic graph
    B = random_cyclic_graph(7; rng = MersenneTwister(5))
    gap = maximum(abs.(implied_total_effects(B) .- inv(Matrix{Float64}(I, 7, 7) - B)))
    sayf("  cyclic: |T - (I-B)^-1|                      %.3e (expected > 0)\n", gap)
    pass("differs from (I-B)^-1 when cyclic", gap > 1e-3)
end

# ---------------------------------------------------------------------------
# 2. analytic gradient vs central finite differences
# ---------------------------------------------------------------------------
say("=== 2. analytic gradient ===")
function toy_model(p; seed = 1, se = 0.05, kwargs...)
    rng = MersenneTwister(seed)
    B = random_cyclic_graph(p; rng = rng)
    Θ = implied_total_effects(B) - Matrix{Float64}(I, p, p) .+ se .* randn(rng, p, p)
    for u in 1:p; Θ[u, u] = 0.0; end
    active = trues(p, p); for u in 1:p; active[u, u] = false; end
    te = TotalEffectEstimate(Θ, fill(se, p, p), nothing, fill(-1.0, p), active,
                             fill(20, p), fill(15, p), collect(1:p), ["gene_$i" for i in 1:p])
    return llcb_model(te; kwargs...), B
end

function fd_grad(model, θ; h = 1e-6)
    g = similar(θ)
    for i in eachindex(θ)
        a = copy(θ); a[i] += h
        b = copy(θ); b[i] -= h
        g[i] = (llcb_logdensity(model, a) - llcb_logdensity(model, b)) / (2h)
    end
    return g
end

let
    rng = MersenneTwister(7)
    worst = 0.0
    for prior in (:normal, :studentt, :horseshoe), disp in (false, true)
        model, Btrue = toy_model(6; prior = prior, estimate_dispersion = disp)
        θ = initial_parameters(model; B0 = Btrue .+ 0.05 .* randn(rng, 6, 6))
        θ .+= 0.1 .* randn(rng, length(θ))
        lp, ga = llcb_logdensity_and_gradient(model, θ)
        gf = fd_grad(model, θ)
        rel = norm(ga - gf) / max(norm(gf), 1e-12)
        worst = max(worst, rel)
        sayf("  prior=%-10s phi=%-5s dim=%5d logp=%11.3f rel-err=%.3e\n",
                prior, disp ? "free" : "fixed", length(θ), lp, rel)
    end
    # stability penalty branch
    model, Btrue = toy_model(6; prior = :normal, rho_max = 0.2, stab_scale = 0.05)
    θ = initial_parameters(model; B0 = 1.5 .* Btrue)
    σ, = _sigma_max(to_matrix(view(θ, 1:model.m), 6))
    lp, ga = llcb_logdensity_and_gradient(model, θ)
    gf = fd_grad(model, θ)
    rel = norm(ga - gf) / max(norm(gf), 1e-12)
    worst = max(worst, rel)
    sayf("  stability active (sigma=%.3f > 0.2)      rel-err=%.3e\n", σ, rel)
    pass("gradient matches finite differences", worst < 1e-4)

    # determinism: the published spectral_radius seeded power iteration with rand()
    vals = unique([llcb_logdensity(model, θ) for _ in 1:8])
    sayf("  distinct log densities over 8 evaluations   %d\n", length(vals))
    pass("log density is deterministic", length(vals) == 1)
end

# ---------------------------------------------------------------------------
# 3. stage 1: donor-adjusted Wald ratio vs the published pooled OLS
# ---------------------------------------------------------------------------
say("=== 3. stage-1 total effects ===")
function legacy_total_effects(X, ivm, controls)
    p = size(X, 2)
    x = X .- mean(view(X, controls, :); dims = 1)
    sds = vec(std(view(X, controls, :); dims = 1)); sds[sds .<= 0] .= 1.0
    x = x ./ sds'
    T = zeros(p, p)
    for i in 1:p
        obs = findall((view(ivm, :, i) .== 1) .| collect(controls))
        xi = x[obs, i]
        A = hcat(ones(length(obs)), xi)
        for j in 1:p
            i == j && continue
            T[i, j] = (A \ x[obs, j])[2]
        end
    end
    return T
end

let
    rng = MersenneTwister(2024)
    p = 8
    rows_new = Float64[]; rows_old = Float64[]
    strong_new = Float64[]; strong_old = Float64[]; nweak = 0
    weak_new = Float64[]; weak_old = Float64[]
    cover = Int[]; ncov = 0
    for sim in 1:12
        B = random_cyclic_graph(p; rng = rng)
        X, ivm, donor, controls = simulate_experiment(
            B; n_donors = 3, n_replicates = 8, donor_sd = 0.4,
            donor_imbalance = 0.8, noise_sd = 0.15, rng = rng)
        sds = vec(std(view(X, controls, :); dims = 1))
        Bs = B .* (sds * (1 ./ sds)')
        Ttrue = implied_total_effects(Bs) - Matrix{Float64}(I, p, p)

        te = estimate_total_effects(X, ivm, donor, controls; verbose = false)
        old = legacy_total_effects(X, ivm, controls)

        idx = [(u, j) for u in 1:p, j in 1:p if u != j]
        t = [Ttrue[u, j] for (u, j) in idx]
        hn = [te.Θ[u, j] for (u, j) in idx]
        ho = [old[u, j] for (u, j) in idx]
        push!(rows_new, sqrt(mean((hn .- t) .^ 2)))
        push!(rows_old, sqrt(mean((ho .- t) .^ 2)))
        # rows with an identifiable knockdown (first-stage F >= 10)
        wkrows = Set(weak_knockdowns(te)); nweak += length(wkrows)
        st = [!(u in wkrows) for (u, j) in idx]
        push!(strong_new, sqrt(mean((hn[st] .- t[st]) .^ 2)))
        push!(strong_old, sqrt(mean((ho[st] .- t[st]) .^ 2)))
        wk = abs.(t) .< 0.05
        if any(wk)
            push!(weak_new, sqrt(mean((hn[wk] .- t[wk]) .^ 2)))
            push!(weak_old, sqrt(mean((ho[wk] .- t[wk]) .^ 2)))
        end
        # interval coverage of the delta-method standard errors
        for (u, j) in idx
            se = te.se[u, j]
            (isfinite(se) && se > 0) || continue
            ncov += 1
            abs(te.Θ[u, j] - Ttrue[u, j]) <= 1.96 * se && (push!(cover, 1))
        end
    end
    sayf("  RMSE  (all entries)   legacy %.4f   new %.4f   (new wins %d of %d datasets)\n",
         mean(rows_old), mean(rows_new), count(rows_new .< rows_old), length(rows_new))
    sayf("  RMSE  (F >= 10 rows)  legacy %.4f   new %.4f   (%d weak knockdowns excluded)\n",
         mean(strong_old), mean(strong_new), nweak)
    sayf("  RMSE  (weak entries)  legacy %.4f   new %.4f\n", mean(weak_old), mean(weak_new))
    sayf("  95%% SE coverage of the new estimator: %.3f over %d entries\n",
            length(cover) / ncov, ncov)
    # The all-entries mean is reported but not tested: weak knockdowns give the
    # Wald ratio heavy tails, so its mean depends on whether one appears in the
    # 12 draws (it flips between Julia versions' RNG streams). See bench/stage1_study.jl.
    pass("new stage 1 beats pooled OLS in most datasets", count(rows_new .< rows_old) > length(rows_new) / 2)
    pass("new stage 1 beats pooled OLS on identified knockdowns", mean(strong_new) < mean(strong_old))
    pass("new stage 1 beats pooled OLS on weak edges", mean(weak_new) < mean(weak_old))
    pass("stage-1 standard errors roughly calibrated", 0.85 <= length(cover) / ncov <= 1.0)
end

# ---------------------------------------------------------------------------
# 4. threaded stage 1 must equal serial stage 1
# ---------------------------------------------------------------------------
say("=== 4. threaded vs serial stage 1 (", Threads.nthreads(), " threads) ===")
let
    rng = MersenneTwister(77)
    p = 12
    B = random_cyclic_graph(p; rng = rng)
    X, ivm, donor, controls = simulate_experiment(B; n_replicates = 6, rng = rng)
    ref = estimate_total_effects(X, ivm, donor, controls; verbose = false, threads = false)
    mism = 0
    for _ in 1:200
        te = estimate_total_effects(X, ivm, donor, controls; verbose = false, threads = true)
        (te.active == ref.active && te.Θ == ref.Θ && te.se == ref.se) || (mism += 1)
    end
    sayf("  threaded runs differing from serial: %d of 200 (active moments in serial run: %d of %d)\n",
         mism, count(ref.active), p * (p - 1))
    # demonstrate the failure mode the fix removes: concurrent writes to
    # different rows of one BitMatrix column
    racy = 0
    for _ in 1:200
        A = falses(p, p)
        Threads.@threads for k in 1:p
            for _ in 1:2000; A[k, :] .= true; A[k, k] = false; end
        end
        count(A) == p * (p - 1) || (racy += 1)
    end
    sayf("  (control) BitMatrix written row-wise from threads lost bits in %d of 200 trials\n", racy)
    Threads.nthreads() > 1 || say("  NOTE: run with -t N > 1 for this check to be meaningful")
    pass("threaded stage 1 is deterministic", mism == 0)
end

say("")
nfail = count(!, values(RESULTS))
say(nfail == 0 ? "CORE_VALIDATION_PASS" : "CORE_VALIDATION_FAIL ($nfail failing)")
close(OUT)
