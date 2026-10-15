# InferCausalGraph

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://weinstockj.github.io/LLCB/dev)
[![CI](https://github.com/weinstockj/LLCB/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/weinstockj/LLCB/actions/workflows/ci.yml)

This repo contains the implementation of the 'LLCB' method, which is described
in Weinstock & Arce et al., *Cell Genomics* 2024
([doi:10.1016/j.xgen.2024.100671](https://doi.org/10.1016/j.xgen.2024.100671)).

Version 0.3.0 is a breaking release that requires Julia 1.10 or later. It
replaces the estimation engine and ports the published Turing-based stage 2
to current Turing as `fit_cyclic_model_legacy`. The exact code used in the
manuscript remains available at tag `v0.2.0b` (Julia 1.9.4); see
[Reproducing the published analysis](#reproducing-the-published-analysis),
[What changed in 0.3](#what-changed-in-03), and `NEWS.md`.

## Quick-start

Assuming that you have expression data properly formatted into a dataframe
called `expression`:

```julia
using InferCausalGraph

graph = interventionGraph(expression)
fit, model, total_effects = fit_cyclic_model(graph)
edges = parse_llcb_fit(fit)
```

`edges` is a long-format table with one row per ordered gene pair:

| column | meaning |
|---|---|
| `row`, `col` | parent and child gene |
| `estimate`, `std` | posterior mean and standard deviation of the direct effect |
| `2.5%`, `97.5%` | posterior credible interval |
| `lsfr` | local sign false rate, `min(P(B>0), P(B<0))` — the threshold-free edge evidence |
| `PIP` | `P(|B| > pip_threshold)`, default threshold `0.05`, for comparability with the published output |
| `rhat`, `ess` | per-edge convergence diagnostics |
| `map` | MAP estimate |

Prefer `lsfr` for ranking edges: unlike `PIP` it needs no effect-size cut.

The two stages can also be run separately, which is useful when exploring
priors without repeating stage 1:

```julia
te    = estimate_total_effects(graph)                 # stage 1
model = llcb_model(te; prior = :studentt)             # stage 2 log-posterior
fit, _ = fit_llcb(te; model = model, nchains = 4)
```

## What changed in 0.3

- **Stage 1** estimates total effects as donor-adjusted interventional
  contrasts, with standard errors. It replaces pooled OLS slopes, which mixed
  in within-control (confounded) covariance.
- **Weak knockdowns** (first-stage F < 10) give heavy-tailed total effects.
  `estimate_total_effects` warns about them, and `weak_knockdowns(te)` lists
  them. A knockdown can be weak because the guide is inefficient, or because
  the gene is barely expressed in controls.
- **Stage 2** uses the exact hard-intervention identity
  `T̃ = Diagonal(diag(A))⁻¹ A` with `A = (I − B)⁻¹`. This removes the
  errors-in-variables attenuation of the published `t = T β` formulation, and
  the stage-1 standard errors replace the free noise scale. The dispersion `φ`
  is fixed at 1 by default because no residual degrees of freedom can
  estimate it.
- **The log density is deterministic with an analytic gradient.** The
  published stability penalty drew random numbers inside the model. Smooth
  shrinkage priors (`:studentt`, `:horseshoe`, `:normal`) replace the L1
  penalty.
- **The posterior comes from NUTS**, preconditioned by Pathfinder's
  low-rank-plus-diagonal covariance. The published code reported Pathfinder's
  approximation as the posterior.
- **Memory grows as p² rather than p⁴.** The published dense `T` needed 64 GB
  at 300 genes; the new model state is 1.4 MB, and one gradient takes about
  13 ms.
- **A closed-form estimator** (`closed_form_coefficients`,
  `fit_llcb_closed_form`) inverts the identity exactly when every readout is
  perturbed. It gives a fast first look and the MAP starting point, but it is
  sensitive to weak knockdowns.
- **Threading:** chains run on Julia threads, with BLAS pinned to one thread
  while they sample.

Derivations, design choices and the simulation results are in
[`docs/src/methods.md`](docs/src/methods.md).

## Performance against the published method

Six random cyclic 8-gene datasets with donor-confounded knockouts
(`bench/headtohead.jl`, Julia 1.13.1, 8 threads). "Published" is
`fit_cyclic_model_legacy` with the manuscript defaults, i.e. both of its
stages:

| method | mean AP | RMSE | 95% coverage | median time |
|---|---|---|---|---|
| published (ported) | 0.460 | 0.292 | 0.610 | ~18 s |
| `fit_llcb` (NUTS, 4 chains) | 0.948 | 0.122 | 0.958 | ~5 s |
| `fit_llcb_closed_form` (5 of 6 applicable) | 0.952 | 0.225 | 0.986 | < 2 s |

On six 10-gene datasets, `fit_llcb` converged (`rhat < 1.05`) on all six;
see [methods](docs/src/methods.md#convergence-in-simulation).

## Choosing the prior

| `prior` | sampled parameters | when to use |
|---|---|---|
| `:studentt` (default) | `p(p−1) + 1` | the scalable default; heavy-tailed shrinkage without auxiliary variables |
| `:horseshoe` | `2p(p−1) + 3` | regularised horseshoe (Piironen & Vehtari 2017), non-centred; stronger sparsity at small `p`, but doubles the dimension |
| `:normal` | `p(p−1) + 1` | ridge-like; mainly a reference |

With every readout perturbed, stage 2 is just-identified, so all sparsity comes
from the prior ([details](docs/src/methods.md#identifiability)).

## Installation

Requires Julia 1.10 (the LTS) or later. Tested on 1.10.12 and 1.13.1. The
package is not in the General registry, so install it from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/weinstockj/LLCB", rev = "v0.3.0")
```

or, from a clone:

```julia
using Pkg
Pkg.instantiate()   # only needs to be run once
using InferCausalGraph
```

## Input data formatting

The input variable `expression` should be a dataframe where the rows indicate
the sample. Columns should include the identity of the sample `donor`, the
intervention performed `intervention`, and then an additional column for each
readout gene, where the column names are the gene symbols. Overall, the
dataframe should have one row per sample, and P + 2 columns, where P is the
number of readout genes. The perturbed gene's own column is set to 0 in its
knockout samples.

For an example, run the following from the test helpers:

```julia
include("test/extended_test.jl")
sim_cyclic_expression(cyclic_chain_graph(), 3, 50, true)
```

This simulates expression data for a 5 gene graph with each KO performed in 3
'donors' with 50 replicates. Control data (no KO's) are also included.

The first ten rows of the data looks like:

```
 Row │ gene_1   gene_2   gene_3   gene_4   gene_5   intervention  donor
     │ Float64  Float64  Float64  Float64  Float64  String        String
─────┼───────────────────────────────────────────────────────────────────
   1 │ 0.0      3.36514  4.67425  4.92861  5.44028  gene_1        1
   2 │ 0.0      2.63952  3.93181  4.63513  4.89527  gene_1        1
   3 │ 0.0      2.9394   4.14463  4.9676   5.48139  gene_1        1
```

`bench/simulate.jl` provides a generator that also applies donor effects to
the expression values, and that can make donor composition differ across knockouts, which
is what the stage-1 comparison needs.

## Input data notes

In our manuscript, we used bulk RNA-seq data as the read-out. Our processing
pipeline is available [here](https://github.com/weinstockj/RNAseq-perturbation-CD4-pipeline).

Please note that we analyzed the DESeq2 variance stabilizing transform (`vst`)
function to normalize the data. We also recommend estimating expression PCs and
regressing out those which correspond to unwanted global sources of variation.

## Reproducing the published analysis

For bit-for-bit reproduction, check out tag `v0.2.0b` and use Julia 1.9.4 with
its `Manifest.toml`. Version 0.3.0 also ports the published stage 2 to current
Turing, using the same initialisation and optimizer settings as v0.2.0b (see
[methods](docs/src/methods.md#the-ported-published-model)).
Expect statistically equivalent, not identical, numbers.

```julia
model_pars    = get_model_params(false, 1.0, 0.01)
sampling_pars = get_sampling_params(false)
chain = fit_cyclic_model_legacy(graph, false, model_pars, sampling_pars)
edges = parse_cyclic_chain(chain[1], chain[2], get_cyclic_matrices(graph, false)[3])
```

`estimate_total_effects_legacy` and `get_cyclic_matrices` are likewise retained.

## Tests and benchmarks

```julia
Pkg.test()                                   # fast core tests
LLCB_RUN_SLOW_TESTS=1 julia --project -e 'using Pkg; Pkg.test()'
```

`bench/` holds the scripts behind the claims above: `validate_core.jl`
(identity, gradient, stage-1 accuracy — standard library only),
`scaling.jl` (memory and timing versus gene count), and `benchmark.jl`
(accuracy and calibration with the two changes separated).

## Contact
Please contact Josh Weinstock <josh.weinstock@emory.edu> with questions.

## Citation
If you use this method, please cite:

Weinstock JS, Arce MM, Freimer JW, Ota M, Marson A, Battle A, Pritchard JK.
Gene regulatory network inference from CRISPR perturbations in primary CD4⁺ T
cells elucidates the genomic basis of immune disease. *Cell Genomics*
4(11):100671 (2024). https://doi.org/10.1016/j.xgen.2024.100671
