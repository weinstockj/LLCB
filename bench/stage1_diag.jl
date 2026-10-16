
# Stage-1 estimator study: donor-adjusted Wald ratio vs published pooled OLS,
# over a grid of donor imbalance and replication, many simulations per cell.
using LinearAlgebra, Statistics, Random, Printf
include(joinpath(@__DIR__, "..", "src", "total_effects.jl"))
include(joinpath(@__DIR__, "..", "src", "moment_model.jl"))
const OUT = open(joinpath(@__DIR__, "stage1_diag_output.txt"), "w")
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


p = 8
imb, nrep = 0.0, 10
best = (0.0, 0)
for s in 1:100
    rng = MersenneTwister(hash((imb, nrep, s)))
    B = random_cyclic_graph(p; rng); X, ivm, dn, ctrl = simulate(B; n_rep = nrep, imb, rng)
    sds = vec(std(X[ctrl, :]; dims = 1)); Tt = implied_total_effects(B .* (sds * (1 ./ sds)')) - I
    te = estimate_total_effects(X, ivm, dn, ctrl; verbose = false)
    r = sqrt(mean([(te.Θ[u, j] - Tt[u, j])^2 for u in 1:p, j in 1:p if u != j]))
    r > best[1] && (global best = (r, s))
end
s = best[2]
rng = MersenneTwister(hash((imb, nrep, s)))
B = random_cyclic_graph(p; rng); X, ivm, dn, ctrl = simulate(B; n_rep = nrep, imb, rng)
sds = vec(std(X[ctrl, :]; dims = 1)); Bs = B .* (sds * (1 ./ sds)'); Tt = implied_total_effects(Bs) - I
te = estimate_total_effects(X, ivm, dn, ctrl; verbose = false); old = legacy(X, ivm, ctrl)
sayf("worst dataset s = %d, new RMSE %.3f\n", s, best[1])
sayf("spectral radius of B %.3f; smallest singular value of (I - B) %.4f\n", maximum(abs.(eigvals(B))), minimum(svdvals(I - B)))
sayf("control SDs: %s\n", string(round.(sds; digits = 4)))
sayf("knockdown (scaled units) per KO: %s\n", string(round.(te.knockdown; digits = 2)))
for u in 1:p
    keep = setdiff(1:p, u)
    sayf("  KO %d: min singular value of I - B[-u,-u] = %.4f; row max |true T| = %.2f\n", u, minimum(svdvals(I - B[keep, keep])), maximum(abs.(Tt[u, keep])))
end
errs = sort([(abs(te.Θ[u, j] - Tt[u, j]), u, j) for u in 1:p, j in 1:p if u != j]; rev = true)
say("largest errors (u -> j): true | new (se) | old")
for (e, u, j) in errs[1:8]
    sayf("  %d -> %d: %8.3f | %8.3f (%.3f) | %8.3f\n", u, j, Tt[u, j], te.Θ[u, j], te.se[u, j], old[u, j])
end
close(OUT)
