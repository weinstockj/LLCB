
# =============================================================================
# Simulation harness shared by the tests and the benchmark.
#
# Extends the simulator in test/extended_test.jl in two ways that matter for
# what is being evaluated here:
#   * donor effects are applied to the expression values (`sim_cyclic_expression`
#     draws `ω` but does not add it), and their composition can be made to
#     differ across knockouts so that donor is a genuine confounder;
#   * arbitrary sparse cyclic graphs at arbitrary size, not just the 5-gene
#     chain.
# =============================================================================

using Random, LinearAlgebra, Statistics, DataFrames, Distributions

"""
    random_cyclic_graph(p; n_edges, strength, rho, rng) -> Matrix

Random sparse directed graph with cycles. `B[i, j]` is the direct effect of
parent `i` on child `j`. Edge weights are drawn uniformly from `±strength` and
the whole matrix is rescaled so that its spectral radius equals `rho`, which
keeps `(I - B)⁻¹` well conditioned.
"""
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

"""
    cyclic_chain(p; weight) -> Matrix

A single directed cycle `1 → 2 → ... → p → 1`, the graph used in the repo's
own test (`cyclic_chain_graph`) generalised to `p` nodes.
"""
function cyclic_chain(p::Int; weight::Float64 = 0.5)
    B = zeros(p, p)
    for i in 1:p
        B[i, mod1(i + 1, p)] = weight
    end
    return B
end

"""
    simulate_experiment(B; kwargs...) -> (DataFrame, NamedTuple)

Simulate a knockout panel from the linear cyclic system `x = xB + e`, in which
a knockout of gene `k` is a hard intervention setting `x_k = 0`.

# Keyword arguments
- `n_donors = 3`, `n_replicates = 10`: replicates per donor per condition.
- `donor_sd = 0.3`: standard deviation of donor offsets applied to the
  exogenous term, so they propagate through the network.
- `donor_imbalance = 0.0`: in `[0, 1]`. At 0 every condition uses every donor
  equally; at 1 each knockout is run in a single donor, making donor a
  confounder of the knockout contrast.
- `noise_sd = 0.1`, `baseline = 1.0`: scale and mean of the exogenous term.
- `perturbed = 1:p`: which genes are knocked out.
- `include_controls = true`.

Returns the expression `DataFrame` in the layout `interventionGraph` expects,
and a `NamedTuple` with the true `B`, the implied total effects, and the
control standard deviations needed to put `B` on the scaled axis that
`estimate_total_effects(; scale = true)` targets.
"""
function simulate_experiment(
        B::Matrix{Float64};
        n_donors::Int = 3,
        n_replicates::Int = 10,
        donor_sd::Float64 = 0.3,
        donor_imbalance::Float64 = 0.0,
        noise_sd::Float64 = 0.1,
        baseline::Float64 = 1.0,
        perturbed = 1:size(B, 1),
        include_controls::Bool = true,
        rng = Random.GLOBAL_RNG,
    )
    p = size(B, 1)
    perturbed = collect(perturbed)
    donor_offset = donor_sd .* randn(rng, n_donors, p)

    conditions = include_controls ? vcat(perturbed, 0) : perturbed
    rows = Vector{Vector{Float64}}()
    intervention = String[]
    donor = String[]

    for k in conditions
        Bk = copy(B)
        k > 0 && (Bk[:, k] .= 0.0)
        Ak = inv(Matrix{Float64}(I, p, p) - Bk)
        # donor composition for this condition
        w = fill(1.0 / n_donors, n_donors)
        if k > 0 && donor_imbalance > 0
            focus = mod1(k, n_donors)
            w .= (1 - donor_imbalance) / n_donors
            w[focus] += donor_imbalance
        end
        total = n_donors * n_replicates
        counts = max.(1, round.(Int, w .* total))
        for d in 1:n_donors, _ in 1:counts[d]
            e = baseline .+ view(donor_offset, d, :) .+ noise_sd .* randn(rng, p)
            k > 0 && (e[k] = 0.0)
            x = vec(e' * Ak)
            k > 0 && (x[k] = 0.0)
            push!(rows, x)
            push!(intervention, k > 0 ? "gene_$k" : "control")
            push!(donor, string(d))
        end
    end

    X = reduce(vcat, transpose.(rows))
    df = DataFrame(X, [Symbol("gene_$i") for i in 1:p])
    df.intervention = intervention
    df.donor = donor

    A = inv(Matrix{Float64}(I, p, p) - B)
    T = A ./ diag(A)
    ctrl = intervention .== "control"
    sds = include_controls ? vec(std(X[ctrl, :]; dims = 1)) : vec(std(X; dims = 1))
    B_scaled = B .* (sds * (1 ./ sds)')      # B[i,j] σ_i / σ_j
    return df, (B = B, B_scaled = B_scaled, total_effects = T, sds = sds, perturbed = perturbed)
end
