
# Unit tests for the LLCB numerical core and the stage-1 estimator.
using InferCausalGraph
using InferCausalGraph: implied_total_effects, llcb_model, llcb_logdensity,
                        llcb_logdensity_and_gradient, llcb_dimension,
                        initial_parameters, to_matrix, from_matrix, free_indices,
                        TotalEffectEstimate, estimate_total_effects,
                        estimate_total_effects_legacy, linear_regress,
                        spectral_radius, interventionGraph, fit_llcb,
                        parse_llcb_fit, dispersion_index, closed_form_coefficients,
                        has_full_total_effects, fit_llcb_closed_form,
                        weak_knockdowns
using Test, LinearAlgebra, Statistics, Random, DataFrames

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
function _random_cyclic_graph(p::Int; n_edges::Int = 2p, rho::Float64 = 0.6,
                              rng = Random.GLOBAL_RNG)
    B = zeros(p, p)
    offdiag = [(i, j) for i in 1:p, j in 1:p if i != j]
    for s in randperm(rng, length(offdiag))[1:min(n_edges, length(offdiag))]
        i, j = offdiag[s]
        B[i, j] = (rand(rng, Bool) ? 1 : -1) * (0.2 + 0.4 * rand(rng))
    end
    ρ = maximum(abs.(eigvals(B)))
    ρ > 0 && (B .*= rho / ρ)
    return B
end

function _simulate(B; n_donors = 3, n_replicates = 10, donor_sd = 0.4,
                   donor_imbalance = 0.6, noise_sd = 0.15, rng = Random.GLOBAL_RNG)
    p = size(B, 1)
    off = donor_sd .* randn(rng, n_donors, p)
    rows = Vector{Vector{Float64}}(); interv = Int[]; donor = Int[]
    for k in vcat(1:p, 0)
        Bk = copy(B); k > 0 && (Bk[:, k] .= 0.0)
        Ak = inv(Matrix{Float64}(I, p, p) - Bk)
        w = fill(1.0 / n_donors, n_donors)
        if k > 0 && donor_imbalance > 0
            w .= (1 - donor_imbalance) / n_donors
            w[mod1(k, n_donors)] += donor_imbalance
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
    return X, ivm, donor, interv .== 0, interv
end

function _toy_estimate(p; se = 0.05, seed = 1)
    rng = MersenneTwister(seed)
    B = _random_cyclic_graph(p; rng = rng)
    Θ = implied_total_effects(B) - Matrix{Float64}(I, p, p) .+ se .* randn(rng, p, p)
    for u in 1:p; Θ[u, u] = 0.0; end
    active = trues(p, p); for u in 1:p; active[u, u] = false; end
    te = TotalEffectEstimate(Θ, fill(se, p, p), nothing, fill(-1.0, p), active,
                             fill(20, p), fill(15, p), collect(1:p),
                             ["gene_$i" for i in 1:p])
    return te, B
end

# ---------------------------------------------------------------------------
# the exact hard-intervention total-effect identity
# ---------------------------------------------------------------------------
function test_moment_identity()
    rng = MersenneTwister(11)
    for _ in 1:10
        p = 7
        B = _random_cyclic_graph(p; rng = rng)
        T = implied_total_effects(B)

        # self-effects are normalised to one
        @test all(isapprox.(diag(T), 1.0; atol = 1e-12))

        # satisfies "total effect on v = sum over parents of (total effect on
        # the parent) x (direct effect)", the recursion LLCB encodes
        for u in 1:p, v in 1:p
            u == v && continue
            @test isapprox(T[u, v], sum(T[u, w] * B[w, v] for w in 1:p if w != v);
                           atol = 1e-10)
        end

        # agrees with solving each hard intervention separately
        for u in 1:p
            keep = setdiff(1:p, u)
            brute = vec(B[u, keep]' * inv(Matrix{Float64}(I, p - 1, p - 1) - B[keep, keep]))
            @test isapprox(brute, T[u, keep]; atol = 1e-10)
        end
    end

    # reduces to (I - B)^-1 on a DAG, and differs from it when cyclic
    p = 7
    Bdag = 0.3 .* triu(abs.(_random_cyclic_graph(p; rng = MersenneTwister(3))), 1)
    @test isapprox(implied_total_effects(Bdag),
                   inv(Matrix{Float64}(I, p, p) - Bdag); atol = 1e-10)
    Bcyc = _random_cyclic_graph(p; rng = MersenneTwister(5))
    @test maximum(abs.(implied_total_effects(Bcyc) -
                       inv(Matrix{Float64}(I, p, p) - Bcyc))) > 1e-3
end

# ---------------------------------------------------------------------------
# analytic gradient
# ---------------------------------------------------------------------------
function _fd_gradient(model, θ; h = 1e-6)
    g = similar(θ)
    for i in eachindex(θ)
        a = copy(θ); a[i] += h
        b = copy(θ); b[i] -= h
        g[i] = (llcb_logdensity(model, a) - llcb_logdensity(model, b)) / (2h)
    end
    return g
end

function test_analytic_gradient()
    rng = MersenneTwister(7)
    te, Btrue = _toy_estimate(6)
    for prior in (:normal, :studentt, :horseshoe), disp in (false, true)
        model = disp ? (@test_logs (:warn,) match_mode = :any llcb_model(te; prior = prior, estimate_dispersion = true)) :
                       llcb_model(te; prior = prior)
        θ = initial_parameters(model; B0 = Btrue .+ 0.05 .* randn(rng, 6, 6))
        θ .+= 0.1 .* randn(rng, length(θ))
        _, ga = llcb_logdensity_and_gradient(model, θ)
        gf = _fd_gradient(model, θ)
        @test norm(ga - gf) / max(norm(gf), 1e-12) < 1e-4
        # the parameter vector must have exactly the advertised length
        @test length(θ) == llcb_dimension(model)
        @test dispersion_index(model) == (disp ? length(θ) : 0)
    end
    # the default fixes φ: the LLCB design has no residual degrees of freedom
    @test !llcb_model(te).estimate_dispersion

    # the stability-penalty branch must be differentiated too
    model = llcb_model(te; prior = :normal, rho_max = 0.2, stab_scale = 0.05)
    θ = initial_parameters(model; B0 = 1.5 .* Btrue)
    _, ga = llcb_logdensity_and_gradient(model, θ)
    gf = _fd_gradient(model, θ)
    @test norm(ga - gf) / max(norm(gf), 1e-12) < 1e-4
end

function test_closed_form()
    rng = MersenneTwister(17)
    for p in (5, 12, 40)
        B = _random_cyclic_graph(p; rng = rng)
        @test maximum(abs.(closed_form_coefficients(implied_total_effects(B)) .- B)) < 1e-10
    end
    # it is the flat-prior optimum: it reproduces the observed total effects
    # exactly, so the moment residual vanishes there
    te, _ = _toy_estimate(6)
    @test implied_total_effects(closed_form_coefficients(te)) ≈ te.Θ + I atol = 1e-10
    # and it refuses when some readout was never perturbed
    te_short = TotalEffectEstimate(te.Θ[1:5, :], te.se[1:5, :], nothing, te.knockdown[1:5],
                                   te.active[1:5, :], te.nobs[1:5], te.dof[1:5], collect(1:5), te.targets)
    @test !has_full_total_effects(te_short)
    @test_throws ArgumentError closed_form_coefficients(te_short)
end

function test_log_density_determinism()
    te, Btrue = _toy_estimate(6)
    model = llcb_model(te; prior = :studentt)
    θ = initial_parameters(model; B0 = Btrue)
    vals = [llcb_logdensity(model, θ) for _ in 1:8]
    @test length(unique(vals)) == 1
end

function test_packing_roundtrip()
    p = 9
    B = _random_cyclic_graph(p; rng = MersenneTwister(2))
    @test to_matrix(from_matrix(B), p) ≈ B
    @test length(free_indices(p)) == p * (p - 1)
    @test all(iszero, diag(to_matrix(from_matrix(B), p) - B))
end

# ---------------------------------------------------------------------------
# stage 1
# ---------------------------------------------------------------------------
function test_total_effects_recovery()
    rng = MersenneTwister(99)
    p = 6
    B = _random_cyclic_graph(p; rng = rng)
    # near-noiseless, many replicates: the estimator must recover the truth
    X, ivm, donor, controls, _ = _simulate(B; n_replicates = 80, noise_sd = 0.02,
                                           donor_sd = 0.3, rng = rng)
    sds = vec(std(view(X, controls, :); dims = 1))
    Bs = B .* (sds * (1 ./ sds)')
    Ttrue = implied_total_effects(Bs) - Matrix{Float64}(I, p, p)
    te = estimate_total_effects(X, ivm, donor, controls; verbose = false)
    offd = [(u, j) for u in 1:p, j in 1:p if u != j]
    # rows with an identifiable knockdown must be accurate; a weak knockdown
    # (e.g. a gene whose baseline expression is near zero, so knocking it out
    # changes little) must be flagged and stay within its standard errors
    weak = Set(weak_knockdowns(te))
    strong = [(u, j) for (u, j) in offd if !(u in weak)]
    @test length(weak) <= 1
    @test maximum(abs.([te.Θ[u, j] - Ttrue[u, j] for (u, j) in strong])) < 0.05
    for (u, j) in offd
        u in weak || continue
        @test abs(te.Θ[u, j] - Ttrue[u, j]) < 4 * te.se[u, j]
    end
    @test all(te.active[u, j] for (u, j) in offd)
    @test all(.!diag(te.active))
    @test all(isfinite, te.knockdown)
end

function test_total_effects_beats_pooled_ols()
    rng = MersenneTwister(2024)
    p = 8
    err_new = Float64[]; err_old = Float64[]
    for _ in 1:5
        B = _random_cyclic_graph(p; rng = rng)
        X, ivm, donor, controls, interv = _simulate(B; n_replicates = 8, rng = rng)
        sds = vec(std(view(X, controls, :); dims = 1))
        Ttrue = implied_total_effects(B .* (sds * (1 ./ sds)')) - Matrix{Float64}(I, p, p)

        te = estimate_total_effects(X, ivm, donor, controls; verbose = false)

        df = DataFrame(X, [Symbol("gene_$i") for i in 1:p])
        df.intervention = [k > 0 ? "gene_$k" : "control" for k in interv]
        df.donor = string.(donor)
        for i in 1:nrow(df)
            interv[i] > 0 && (df[i, Symbol("gene_$(interv[i])")] = 0.0)
        end
        g = interventionGraph(df)
        old = estimate_total_effects_legacy(g, false, true, true)

        offd = [(u, j) for u in 1:p, j in 1:p if u != j]
        push!(err_new, sqrt(mean([(te.Θ[u, j] - Ttrue[u, j])^2 for (u, j) in offd])))
        push!(err_old, sqrt(mean([(old[u, j] - Ttrue[u, j])^2 for (u, j) in offd])))
    end
    @test mean(err_new) < mean(err_old)
end

function test_intervention_graph_roundtrip()
    rng = MersenneTwister(31)
    p = 5
    B = _random_cyclic_graph(p; rng = rng)
    X, ivm, donor, controls, interv = _simulate(B; n_replicates = 6, rng = rng)
    df = DataFrame(X, [Symbol("gene_$i") for i in 1:p])
    df.intervention = [k > 0 ? "gene_$k" : "control" for k in interv]
    df.donor = string.(donor)
    for i in 1:nrow(df)
        interv[i] > 0 && (df[i, Symbol("gene_$(interv[i])")] = 0.0)
    end
    g = interventionGraph(df)
    @test g.nv == p
    @test sum(g.interventions) == count(>(0), interv)
    te = estimate_total_effects(g; verbose = false)
    @test size(te.Θ) == (p, p)
    @test te.targets == ["gene_$i" for i in 1:p]
    @test te.perturbed == collect(1:p)
end

# ---------------------------------------------------------------------------
# legacy reference functions, kept working
# ---------------------------------------------------------------------------
function test_linear_regress_legacy()
    rng = MersenneTwister(1)
    x = randn(rng, 1_000)
    y = 0.5 .* x .+ 0.01 .* randn(rng, 1_000)
    @test abs(linear_regress(x, y) - 0.5) / 0.5 < 0.01
end

function test_spectral_radius_legacy()
    rng = MersenneTwister(1)
    A = randn(rng, 10, 10); A = A + A'
    B = A .* A
    # the published helper returns the spectral radius of the elementwise
    # square, not of its argument
    @test abs(maximum(abs.(eigvals(B))) - spectral_radius(A, 60, 1e-10)) /
          maximum(abs.(eigvals(B))) < 0.01
end

# ---------------------------------------------------------------------------
# slow: full pipeline
# ---------------------------------------------------------------------------
function test_end_to_end_fit()
    rng = MersenneTwister(4242)
    p = 8
    B = _random_cyclic_graph(p; n_edges = 2p, rng = rng)
    X, ivm, donor, controls, _ = _simulate(B; n_replicates = 10, rng = rng)
    sds = vec(std(view(X, controls, :); dims = 1))
    Btrue = B .* (sds * (1 ./ sds)')
    truth = BitMatrix(abs.(Btrue) .> 1e-12)

    te = estimate_total_effects(X, ivm, donor, controls; verbose = false)
    fit, mdl = fit_llcb(te; prior = :studentt, nchains = 2, n_adapt = 400,
                        n_samples = 400, target_accept = 0.9, seed = 7, verbose = false)

    offd = [(i, j) for i in 1:p, j in 1:p if i != j]
    @test maximum([fit.rhat[i, j] for (i, j) in offd]) < 1.1
    @test minimum([fit.ess[i, j] for (i, j) in offd]) > 50
    # the strongest true edges must rank above the median
    strong = [(i, j) for (i, j) in offd if abs(Btrue[i, j]) > 0.3]
    lsfr_strong = [fit.lsfr[i, j] for (i, j) in strong]
    @test median(lsfr_strong) < 0.05
    tbl = parse_llcb_fit(fit)
    @test nrow(tbl) == p * (p - 1)
    @test all(0 .<= tbl.PIP .<= 1)
    @test all(0 .<= tbl.lsfr .<= 0.5)
end
