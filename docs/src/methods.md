# Methods and design notes (v0.3.0)

This page has the derivations, design choices and simulation results behind
the changes summarised in the README. Benchmark scripts are in `bench/`, and
the release notes are in `NEWS.md`.

## Stage 1: interventional total effects

`estimate_total_effects` regresses every readout on `[intercept, knockout
indicator, donor dummies]` once per knockout, then forms the Wald ratio
`Θ[u, j] = Δ[j] / Δ[u]`. The published estimator took the univariate OLS slope
of readout `j` on readout `u` over `KO(u) ∪ controls`, which mixes the
interventional contrast with the *within-control* covariance of `u` and `j` —
the confounded association the perturbation was meant to bypass — and applied
no donor adjustment even though the model carried donor terms. The estimator
also returns delta-method standard errors (95% coverage 0.95 in simulation)
and, optionally, the full within-knockout covariance.

## Weak knockdowns

The gain from the new stage-1 estimator depends on knockdown strength. A Wald ratio divides by the
knockdown effect `Δ[u]`, so when a knockdown barely moves its own gene
(first-stage F < 10, a *weak knockdown*) that row's ratios become heavy-tailed.
Over 100 simulated 8-gene datasets per design cell, the new estimator had lower
total-effect RMSE than the pooled slope in 66–88% of datasets, and its median
RMSE was 27–58% lower. Its mean RMSE, however, was only modestly lower and in
one cell higher (0.357 vs 0.326), because weak knockdowns (6–13% of rows) give
it a heavy tail. Restricted to rows with F ≥ 10, it won in 96–100% of datasets,
with mean RMSE 43–63% lower. `estimate_total_effects` warns about weak
knockdowns, and `weak_knockdowns(te)` lists them. A knockdown can be weak
because the guide is inefficient, or because the gene's control expression is
already near zero, so removing it changes little. Stage 2 weights each row by
its standard error, so weak rows have little influence there.

## Stage 2: the exact total-effect identity

The noisy estimates are no longer used as a design matrix. For a hard
intervention the normalised total-effect matrix is

```
T̃ = Diagonal(diag(A))⁻¹ A,        A = (I − B)⁻¹
```

Row `u` of `A(I − B) = eᵤᵀ` gives `A[u,v] = Σ_w A[u,w] B[w,v]` for `v ≠ u`;
dividing by `A[u,u]` is exactly the parent recursion the published
`get_cyclic_matrices` encoded. Note this is **not** `(I − B)⁻¹` unless the graph
is acyclic: a hard knockout cuts the edges *into* the perturbed gene, so loops
through it must be removed, which is what the row normalisation does. Writing
the moment condition this way puts the model on the left and the data only on
the right, which removes the errors-in-variables attenuation of the original
`t = T β` formulation, and the free per-block `σ` is replaced by the stage-1
sampling variances.

## Dispersion `φ`

A dispersion `φ` multiplying those variances is available
(`estimate_dispersion = true`) but is fixed at 1 by default: the system is
just-identified, so no residual degrees of freedom inform `φ`, and sampling it
creates a funnel. On a 5-gene test, fixing it took `rhat` from 2.18 to 1.006.

## Deterministic log density and analytic gradient

The
published `spectral_radius` seeded power iteration with `Base.rand` *inside* the
model, so repeated evaluations at the same parameters differed slightly, which
adds noise to L-BFGS and NUTS; it also computed the spectral radius of the
elementwise square `W ∘ W`. The stability guard is now
a deterministic, differentiable bound on the largest singular value, and the
non-differentiable `sum(abs.(W); dims = 1)` penalty is replaced by a choice of
smooth shrinkage priors (`:studentt` by default, `:horseshoe`, `:normal`). The
gradient of the whole posterior is written in closed form — differentiating the
matrix inverse gives `Aᵀ G Aᵀ` — so no automatic differentiation is involved and
the cost is one factorisation plus two matrix products regardless of how many
coefficients are sampled.

## Sampling: NUTS with a Pathfinder metric

The published
analysis used Pathfinder's Gaussian approximation as the posterior. Here Pathfinder supplies the warm start and,
crucially, its diagonal-plus-low-rank covariance is used as the HMC metric: the
moment likelihood is strongly anisotropic, and with a diagonal metric step-size
adaptation collapses toward zero (in testing, step size ~1e-4, `rhat` ≈ 3, ESS
≈ 5). With the Pathfinder metric the same problem samples at `rhat` ≤ 1.01 with
ESS in the hundreds.

## Memory: no dense `T` matrix

`get_cyclic_matrices` allocated
`zeros(nv*(nv-1), nv*(nv-1))` before sparsifying, so the allocation grew as
`nv⁴`: 784 MB at 100 genes, 4.0 GB at 150, 12.7 GB at 200, 64 GB at 300. The
replacement likelihood stores only the `p × p` total-effect matrix — 1.4 MB of
model state at 300 genes, with one log-density-plus-gradient evaluation costing
13 ms there with one BLAS thread (22 ms with eight; Julia 1.13).

## The closed-form estimator

When every readout is also perturbed, the
identity inverts exactly: `B = I − T̃⁻¹ Diagonal(diag(T̃⁻¹))⁻¹`. This costs one
`p × p` inversion (12 ms at 300 genes, 0.3 s at 2,000 on Julia 1.13) and is the flat-prior
MAP. `fit_llcb` uses it as the optimiser's starting point.
`fit_llcb_closed_form(te)` resamples total effects through it to give
uncertainty without MCMC. It is unweighted, so a single weak knockdown can
distort it badly. On 10-gene simulations its RMSE was 0.082 with no weak
knockdowns, but 1.1–5.3 with two. Use it for a quick first look or for very
large `p`, and check `weak_knockdowns(te)` first.

## Threading

Chains run on Julia threads, and BLAS is pinned to one thread
while they sample (`blas_threads = :auto`) to avoid oversubscription. Its
previous setting is restored afterwards.

## Identifiability

With `p` perturbed genes read out on the same `p` genes there are exactly
`p(p−1)` moment equations and `p(p−1)` coefficients, so stage 2 is
just-identified: all posterior uncertainty comes from stage-1 noise propagated
through the identity, and all sparsity comes from the prior. Reading out more
genes than you perturb is supported (`Θ` may be `q × p` with `q < p`) but makes
the system under-determined, so the prior carries more weight.

## The ported published model

Version 0.3.0 also ports that stage 2 to current Turing.
The port uses the same settings as the published analysis:

1. Each Pathfinder run is initialised from Uniform(−0.03, 0.03). Pass
   `init = :published` to start instead from the moment-based point computed
   in `fit_cyclic_model`.
2. The optimizer is `LBFGS(m = 8)`.

Expect statistically equivalent but not identical numbers: the random streams
and Pathfinder internals have changed since then. The discrete-DAG sampler
(`fit_model`) was not part of the published analysis and has not been ported.

## Convergence in simulation

On six simulated 10-gene datasets (`bench/stability.jl`, Julia 1.13.1),
`fit_llcb` reached `rhat < 1.05` on all six (max 1.015). The minimum ESS was
308, 0.01% of transitions were divergent, 95% coverage was 0.963, and mean AP
was 0.886 against a base rate of 0.222.
