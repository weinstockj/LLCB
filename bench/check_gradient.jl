
# Validation of the analytic gradient in moment_model.jl against central
# finite differences, for every prior, including the stability-penalty branch.
using InferCausalGraph
using InferCausalGraph: LLCBModel, llcb_model, TotalEffectEstimate, to_matrix,
                        from_matrix, initial_parameters, data_term!,
                        implied_total_effects, _sigma_max
using LogDensityProblems, LinearAlgebra, Random, Printf, Statistics

function toy_estimate(p::Int; seed = 1, se = 0.05)
    rng = MersenneTwister(seed)
    B = zeros(p, p)
    for _ in 1:(2p)
        i, j = rand(rng, 1:p), rand(rng, 1:p)
        i != j && (B[i, j] = 0.4 * randn(rng))
    end
    ρ = maximum(abs.(eigvals(B)));  ρ > 0 && (B .*= 0.6 / ρ)
    T = implied_total_effects(B)
    Θ = T - Matrix{Float64}(I, p, p)
    Θ .+= se .* randn(rng, p, p)
    for u in 1:p; Θ[u, u] = 0.0; end
    active = trues(p, p)
    for u in 1:p; active[u, u] = false; end
    te = TotalEffectEstimate(
        Θ, fill(se, p, p), nothing, fill(-1.0, p), active,
        fill(20, p), fill(15, p), collect(1:p), ["gene_$i" for i in 1:p],
    )
    return te, B
end

function fd_gradient(model, θ; h = 1e-6)
    g = similar(θ)
    for i in eachindex(θ)
        θp = copy(θ); θp[i] += h
        θm = copy(θ); θm[i] -= h
        g[i] = (LogDensityProblems.logdensity(model, θp) -
                LogDensityProblems.logdensity(model, θm)) / (2h)
    end
    return g
end

function report(name, ga, gf)
    num = norm(ga - gf)
    den = max(norm(gf), 1e-12)
    @printf("  %-34s rel-err = %.3e   max-abs = %.3e\n", name, num / den,
            maximum(abs, ga - gf))
    return num / den
end

println("=== analytic gradient vs central finite differences ===")
p = 6
te, Btrue = toy_estimate(p)
rng = MersenneTwister(7)
worst = 0.0

for prior in (:normal, :studentt, :horseshoe)
    model = llcb_model(te; prior = prior, rho_max = 0.95, estimate_dispersion = true)
    θ = initial_parameters(model; B0 = Btrue .+ 0.05 .* randn(rng, p, p))
    θ .+= 0.1 .* randn(rng, length(θ))
    lp, ga = LogDensityProblems.logdensity_and_gradient(model, θ)
    gf = fd_gradient(model, θ)
    @printf("prior = %-11s dim = %5d  logp = %12.4f\n", prior, length(θ), lp)
    global worst = max(worst, report("gradient", ga, gf))
end

# stability branch: push B well past rho_max so the penalty is active
model = llcb_model(te; prior = :normal, rho_max = 0.2, stab_scale = 0.05)
θ = initial_parameters(model; B0 = 1.5 .* Btrue)
σ, = _sigma_max(to_matrix(view(θ, 1:model.m), p))
@printf("stability branch active: sigma_max = %.3f (rho_max = 0.2)\n", σ)
lp, ga = LogDensityProblems.logdensity_and_gradient(model, θ)
gf = fd_gradient(model, θ)
worst = max(worst, report("gradient (stability active)", ga, gf))

# determinism: the old spectral_radius seeded power iteration with Base.rand
vals = [LogDensityProblems.logdensity(model, θ) for _ in 1:5]
@printf("log density determinism: %d distinct values over 5 evaluations\n",
        length(unique(vals)))

# recovery of the moment identity: implied_total_effects must satisfy the
# recursion the original code encoded as T beta = t
Bc = copy(Btrue)
T = implied_total_effects(Bc)
resid = 0.0
for u in 1:p, v in 1:p
    u == v && continue
    lhs = T[u, v]
    rhs = sum(T[u, w] * Bc[w, v] for w in 1:p if w != v)
    global resid = max(resid, abs(lhs - rhs))
end
@printf("moment identity max residual = %.3e\n", resid)

println(worst < 1e-5 && resid < 1e-10 ? "GRADIENT_CHECK_PASS" : "GRADIENT_CHECK_FAIL")
