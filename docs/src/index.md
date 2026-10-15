```@meta
CurrentModule = InferCausalGraph
```

# InferCausalGraph

Implementation of the LLCB method of Weinstock & Arce et al., *Cell Genomics*
2024, refactored in version 0.3 for calibrated total-effect estimation and a
scalable sampler.

## Overview

LLCB estimates a matrix of **direct** effects `B` (parents in rows, children in
columns) from perturbation experiments, by matching the **total** effects
implied by `B` to total effects estimated from the data. Fitting proceeds in
two stages.

### Stage 1 — interventional total effects

For each perturbed gene `u`, all readouts are regressed jointly on
`[intercept, knockout indicator, donor dummies]` over the samples in
`KO(u) ∪ controls`. Writing `Δ` for the knockout-indicator coefficients, the
normalised total effect is the Wald ratio `Θ[u, j] = Δ[j] / Δ[u]`, whose
delta-method covariance is returned alongside it.

```@docs
estimate_total_effects
TotalEffectEstimate
perturbed_nodes
weak_knockdowns
schafer_strimmer_shrinkage
```

Knockout rows whose first-stage F statistic falls below 10 are *weak*: their
Wald ratios have heavy-tailed errors. `estimate_total_effects` warns about
them. They carry little weight in `fit_llcb` because their standard errors are
large, but they distort any unweighted use of the total effects.

### Stage 2 — the exact moment model

Under a hard intervention the normalised total-effect matrix is

```math
\tilde{T} = \mathrm{Diagonal}(\mathrm{diag}(A))^{-1} A, \qquad A = (I - B)^{-1}
```

Row ``u`` of ``A(I-B) = e_u^\top`` gives ``A[u,v] = \sum_w A[u,w] B[w,v]`` for
``v \neq u``; dividing by ``A[u,u]`` is exactly the recursion "total effect on
``v`` equals the sum over parents of (total effect on the parent) times (direct
effect)". This is **not** ``(I-B)^{-1}`` unless the graph is acyclic — a hard
knockout cuts the edges into the perturbed gene, so feedback loops through it
must be removed, which is what the row normalisation does.

Because the model sits on the left of the moment condition and the data only on
the right, the noisy total effects never act as a design matrix, which is what
removes the attenuation of the original ``t = T\beta`` formulation.

```@docs
llcb_model
LLCBModel
implied_total_effects
llcb_logdensity_and_gradient
llcb_dimension
dispersion_index
coefficients
closed_form_coefficients
has_full_total_effects
initial_parameters
to_matrix
from_matrix
free_indices
```

### Inference

MAP by L-BFGS on the analytic gradient, then Pathfinder for a warm start and a
diagonal-plus-low-rank metric, then NUTS. The metric matters: the moment
likelihood is strongly anisotropic, and a diagonal metric drives step-size
adaptation toward zero.

```@docs
fit_llcb
fit_llcb_closed_form
LLCBFit
map_estimate
parse_llcb_fit
fit_cyclic_model
```

When every readout was also perturbed, the moment identity can be inverted in
closed form, `B = I − T̃⁻¹ Diagonal(diag(T̃⁻¹))⁻¹`. `fit_llcb` uses that point
to start the MAP, and `fit_llcb_closed_form` resamples stage-1 total effects
through it to get uncertainty without MCMC (one `p × p` inversion per draw).

The dispersion `φ` that rescales the stage-1 standard errors is fixed at 1 by
default. With `q` perturbed genes there are `q(p − 1) ≤ p(p − 1)` moments for
`p(p − 1)` coefficients, so no residual degrees of freedom inform `φ`, and
sampling it creates a funnel that NUTS cannot traverse.

## Edge evidence

`parse_llcb_fit` reports both `lsfr`, the local sign false rate
`min(P(B>0), P(B<0))`, and `PIP`, `P(|B| > pip_threshold)`. Prefer `lsfr`: it
requires no effect-size cut, whereas the published output thresholded at a
fixed `|β| > 0.05`.

## The published method

The published Turing-based stage 2 is ported to current Turing with the same
settings as v0.2.0b: each Pathfinder run is initialised from
Uniform(−0.03, 0.03) and optimised with `LBFGS(m = 8)`. For bit-for-bit reproduction of published numbers, use tag
`v0.2.0b` on Julia 1.9.4.

```@docs
fit_cyclic_model_legacy
joint_cyclic_model
estimate_total_effects_legacy
get_cyclic_matrices
parse_cyclic_chain
get_model_params
get_sampling_params
spectral_radius
```

!!! warning "spectral_radius"
    `spectral_radius(W)` returns the largest eigenvalue of the *elementwise
    square* `W ∘ W`, not of `W`, and seeds power iteration with `Base.rand`, so
    successive calls on the same input differ. It is no longer used in the
    fitting path; the stability guard in `llcb_model` is deterministic and
    differentiable.

## Graph utilities

```@docs
interventionGraph
posterior_graph
topological_sort
```
