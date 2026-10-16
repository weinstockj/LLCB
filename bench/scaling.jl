
# =============================================================================
# Scalability: where the published stage-2 construction runs out of memory, and
# what the replacement costs instead.
#
# The original `get_cyclic_matrices` allocates
#     T = zeros(nv*(nv-1), nv*(nv-1))
# densely before calling `sparse(T)`, so the allocation grows as nv^4. The
# replacement likelihood never forms T: it needs only the p x p total-effect
# matrix and one p x p inverse per gradient.
# =============================================================================
using LinearAlgebra, Statistics, Random, Printf

const OUT = open(joinpath(@__DIR__, "scaling_output.txt"), "w")
say(args...) = (println(OUT, args...); flush(OUT))
sayf(fmt, args...) = (print(OUT, Printf.format(Printf.Format(fmt), args...)); flush(OUT))

include(joinpath(@__DIR__, "..", "src", "total_effects.jl"))
include(joinpath(@__DIR__, "..", "src", "moment_model.jl"))

human(bytes) = bytes < 1e3 ? "$(bytes) B" :
               bytes < 1e6 ? @sprintf("%.1f KB", bytes / 1e3) :
               bytes < 1e9 ? @sprintf("%.1f MB", bytes / 1e6) :
               @sprintf("%.2f GB", bytes / 1e9)

function random_cyclic_graph(p::Int; n_edges::Int = 2p, rho::Float64 = 0.6, rng = Random.GLOBAL_RNG)
    B = zeros(p, p)
    for _ in 1:n_edges
        i, j = rand(rng, 1:p), rand(rng, 1:p)
        i != j && (B[i, j] = 0.4 * randn(rng))
    end
    ρ = maximum(abs.(eigvals(B))); ρ > 0 && (B .*= rho / ρ)
    return B
end

function toy_model(p; se = 0.05, rng = MersenneTwister(1), kwargs...)
    B = random_cyclic_graph(p; rng = rng)
    Θ = implied_total_effects(B) - Matrix{Float64}(I, p, p) .+ se .* randn(rng, p, p)
    for u in 1:p; Θ[u, u] = 0.0; end
    active = trues(p, p); for u in 1:p; active[u, u] = false; end
    te = TotalEffectEstimate(Θ, fill(se, p, p), nothing, fill(-1.0, p), active,
                             fill(20, p), fill(15, p), collect(1:p), ["gene_$i" for i in 1:p])
    return llcb_model(te; kwargs...), B
end

# ---------------------------------------------------------------------------
say("=== 1. memory of the published dense T, by gene count ===")
sayf("%6s %12s %14s %10s\n", "genes", "T is", "dense bytes", "allocates?")
let
    total_ram = Sys.total_memory()
    sayf("# total system memory reported: %s\n", human(total_ram))
    for p in (20, 40, 60, 80, 100, 150, 200, 300)
        n = p * (p - 1)
        bytes = Float64(n)^2 * 8
        ok = "not tried"
        if bytes < 2.0e9
            ok = try
                A = zeros(n, n); s = size(A, 1); A = nothing; GC.gc(); "yes"
            catch e
                "NO ($(typeof(e)))"
            end
        else
            ok = "skipped"
        end
        sayf("%6d %12s %14s %10s\n", p, "$(n)x$(n)", human(bytes), ok)
    end
    say("# entries above 2 GB were not attempted, to avoid thrashing the machine.")
end

say("")
say("=== 2. cost of one log-density + analytic gradient, new model ===")
sayf("%6s %10s %12s %12s %14s\n", "genes", "coef", "model bytes", "grad (ms)", "grad alloc")
let
    for p in (10, 20, 40, 60, 100, 150, 200, 300)
        model, B = toy_model(p; prior = :studentt)
        θ = initial_parameters(model; B0 = B)
        llcb_logdensity_and_gradient(model, θ)              # warm up
        reps = p <= 60 ? 20 : 5
        t = @elapsed for _ in 1:reps
            llcb_logdensity_and_gradient(model, θ)
        end
        alloc = @allocated llcb_logdensity_and_gradient(model, θ)
        mbytes = Base.summarysize(model)
        sayf("%6d %10d %12s %12.3f %14s\n", p, model.m, human(mbytes),
             1000 * t / reps, human(alloc))
    end
end

say("")
say("=== 3. stage 1: published O(p^2) univariate fits vs one solve per knockout ===")
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

function simulate(p; n_replicates = 6, rng = MersenneTwister(3))
    B = random_cyclic_graph(p; rng = rng)
    donor_offset = 0.3 .* randn(rng, 3, p)
    rows = Vector{Vector{Float64}}(); interv = Int[]; donor = Int[]
    for k in vcat(1:p, 0)
        Bk = copy(B); k > 0 && (Bk[:, k] .= 0.0)
        Ak = inv(Matrix{Float64}(I, p, p) - Bk)
        for d in 1:3, _ in 1:n_replicates
            e = 1.0 .+ view(donor_offset, d, :) .+ 0.15 .* randn(rng, p)
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

sayf("%6s %9s %12s %12s %9s\n", "genes", "samples", "legacy (s)", "new (s)", "speedup")
let
    for p in (10, 20, 40, 80, 120)
        X, ivm, donor, controls = simulate(p)
        legacy_total_effects(X, ivm, controls)                       # warm up
        estimate_total_effects(X, ivm, donor, controls; verbose = false)
        t_old = @elapsed legacy_total_effects(X, ivm, controls)
        t_new = @elapsed estimate_total_effects(X, ivm, donor, controls; verbose = false)
        sayf("%6d %9d %12.3f %12.3f %9.1fx\n", p, size(X, 1), t_old, t_new, t_old / t_new)
    end
end

say("")
say("=== 4. gradient cost vs BLAS threads (the p = 60 -> 100 jump) ===")
sayf("%6s %12s %12s %12s\n", "genes", "BLAS 1 (ms)", "BLAS 4 (ms)", "BLAS 8 (ms)")
let
    nb0 = BLAS.get_num_threads()
    for p in (40, 60, 80, 100, 150, 200, 300, 500)
        model, B = toy_model(p; prior = :studentt)
        θ = initial_parameters(model; B0 = B)
        ts = Float64[]
        for nb in (1, 4, 8)
            BLAS.set_num_threads(nb)
            llcb_logdensity_and_gradient(model, θ)
            reps = p <= 100 ? 20 : 5
            t = @elapsed begin
                for _ in 1:reps
                    llcb_logdensity_and_gradient(model, θ)
                end
            end
            push!(ts, 1000 * t / reps)
        end
        sayf("%6d %12.3f %12.3f %12.3f\n", p, ts...)
    end
    BLAS.set_num_threads(nb0)
end

say("")
say("=== 5. closed-form estimator B = I - inv(T) * Diagonal(diag(inv(T)))^-1 ===")
sayf("%6s %12s %14s\n", "genes", "time (ms)", "max |B̂ - B|")
let
    for p in (100, 300, 1000, 2000)
        B = random_cyclic_graph(p; n_edges = 2p, rng = MersenneTwister(p))
        T = implied_total_effects(B)
        closed_form_coefficients(T)
        t = @elapsed (Bh = closed_form_coefficients(T))
        sayf("%6d %12.1f %14.2e\n", p, 1000 * t, maximum(abs.(Bh .- B)))
    end
end

say("")
say("SCALING_DONE")
close(OUT)
