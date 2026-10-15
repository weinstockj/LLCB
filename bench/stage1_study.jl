
# Stage-1 estimator study: donor-adjusted Wald ratio vs published pooled OLS,
# over a grid of donor imbalance and replication, many simulations per cell.
using LinearAlgebra, Statistics, Random, Printf
include(joinpath(@__DIR__, "..", "src", "total_effects.jl"))
include(joinpath(@__DIR__, "..", "src", "moment_model.jl"))
const OUT = open(joinpath(@__DIR__, "stage1_study_output.txt"), "w")
say(a...) = (println(OUT, a...); flush(OUT))
sayf(fmt, a...) = (print(OUT, Printf.format(Printf.Format(fmt), a...)); flush(OUT))

function random_cyclic_graph(p; n_edges = 2p, rho = 0.6, rng)
    B = zeros(p, p); off = [(i, j) for i in 1:p, j in 1:p if i != j]
    for s in randperm(rng, length(off))[1:n_edges]
        i, j = off[s]; B[i, j] = (rand(rng, Bool) ? 1 : -1) * (0.2 + 0.4rand(rng))
    end
    ρ = maximum(abs.(eigvals(B))); ρ > 0 && (B .*= rho / ρ); B
end
function simulate(B; n_donors = 3, n_rep, donor_sd = 0.4, imb, noise_sd = 0.15, rng)
    p = size(B, 1); off = donor_sd .* randn(rng, n_donors, p)
    rows = Vector{Vector{Float64}}(); iv = Int[]; dn = Int[]
    for k in vcat(1:p, 0)
        Bk = copy(B); k > 0 && (Bk[:, k] .= 0); Ak = inv(I - Bk)
        w = fill(1 / n_donors, n_donors)
        if k > 0 && imb > 0; w .= (1 - imb) / n_donors; w[mod1(k, n_donors)] += imb; end
        for d in 1:n_donors, _ in 1:max(1, round(Int, w[d] * n_donors * n_rep))
            e = 1.0 .+ off[d, :] .+ noise_sd .* randn(rng, p); k > 0 && (e[k] = 0)
            x = vec(e' * Ak); k > 0 && (x[k] = 0); push!(rows, x); push!(iv, k); push!(dn, d)
        end
    end
    X = reduce(vcat, transpose.(rows)); ivm = zeros(Int8, size(X, 1), p)
    for i in axes(X, 1); iv[i] > 0 && (ivm[i, iv[i]] = 1); end
    X, ivm, dn, iv .== 0
end
function legacy(X, ivm, ctrl)
    p = size(X, 2); x = (X .- mean(X[ctrl, :]; dims = 1)) ./ std(X[ctrl, :]; dims = 1)
    T = zeros(p, p)
    for i in 1:p
        o = findall((ivm[:, i] .== 1) .| ctrl); A = hcat(ones(length(o)), x[o, i])
        for j in 1:p; i == j || (T[i, j] = (A \ x[o, j])[2]); end
    end
    T
end

p = 8; nsim = 100
say("p = $p, simulations per cell = $nsim, julia $(VERSION)")
sayf("%5s %5s | %-30s | %-15s | %-15s | %-23s | %-23s | %5s\n", "imb", "reps", "RMSE mean (median) old / new", "weak old/new", "strong old/new", "p90 old / new", "max old / new", "wins")
for imb in (0.0, 0.5, 0.8), nrep in (4, 10)
    ro2 = Float64[]; rn2 = Float64[]; nweak = 0; nrows = 0
    ro = Float64[]; rn = Float64[]; wo = Float64[]; wn = Float64[]; so = Float64[]; sn = Float64[]
    for s in 1:nsim
        rng = MersenneTwister(hash((imb, nrep, s)))
        B = random_cyclic_graph(p; rng)
        X, ivm, dn, ctrl = simulate(B; n_rep = nrep, imb, rng)
        sds = vec(std(X[ctrl, :]; dims = 1)); Bs = B .* (sds * (1 ./ sds)')
        Tt = implied_total_effects(Bs) - I
        te = estimate_total_effects(X, ivm, dn, ctrl; verbose = false)
        old = legacy(X, ivm, ctrl)
        idx = [(u, j) for u in 1:p, j in 1:p if u != j]
        t = [Tt[u, j] for (u, j) in idx]; hn = [te.Θ[u, j] for (u, j) in idx]; ho = [old[u, j] for (u, j) in idx]
        wk = abs.(t) .< 0.05
        weakrows = weak_knockdowns(te); nweak += length(weakrows); nrows += p
        strong_rows = [!(u in weakrows) for (u, j) in idx]
        if any(strong_rows)
            push!(ro2, sqrt(mean((ho[strong_rows] .- t[strong_rows]) .^ 2)))
            push!(rn2, sqrt(mean((hn[strong_rows] .- t[strong_rows]) .^ 2)))
        end
        push!(ro, sqrt(mean((ho .- t) .^ 2))); push!(rn, sqrt(mean((hn .- t) .^ 2)))
        any(wk) && (push!(wo, sqrt(mean((ho[wk] .- t[wk]) .^ 2))); push!(wn, sqrt(mean((hn[wk] .- t[wk]) .^ 2))))
        push!(so, sqrt(mean((ho[.!wk] .- t[.!wk]) .^ 2))); push!(sn, sqrt(mean((hn[.!wk] .- t[.!wk]) .^ 2)))
    end
    sayf("%5.1f %5d | %.3f (%.3f) / %.3f (%.3f) | %.3f / %.3f | %.3f / %.3f | %.3f / %.3f | %.3f / %.3f | %3d%%\n", imb, nrep,
         mean(ro), median(ro), mean(rn), median(rn), mean(wo), mean(wn), mean(so), mean(sn),
         quantile(ro, 0.9), quantile(rn, 0.9), maximum(ro), maximum(rn), round(Int, 100 * mean(rn .< ro)))
    sayf("            weak-knockdown rows (F < 10): %.1f%%; excluding them: RMSE mean %.3f / %.3f, median %.3f / %.3f, max %.3f / %.3f, new wins %d%%\n",
         100 * nweak / nrows, mean(ro2), mean(rn2), median(ro2), median(rn2), maximum(ro2), maximum(rn2), round(Int, 100 * mean(rn2 .< ro2)))
    worst = argmax(rn)
    sayf("            worst new-estimator dataset: RMSE %.3f (old %.3f on the same data)\n", rn[worst], ro[worst])
end
say("STUDY_DONE"); close(OUT)
