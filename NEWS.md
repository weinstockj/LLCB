# InferCausalGraph.jl release notes

## v0.3.0 (breaking)

Requires Julia 1.10 or later (tested on 1.10.12 LTS and 1.13.1). The code
used in Weinstock & Arce et al. (*Cell Genomics* 2024) remains at tag
`v0.2.0b` (Julia 1.9.4, Turing 0.29). Check out that tag with its Manifest for
bit-for-bit reproduction.

### Breaking changes

- **Julia and dependencies.** `julia = "1.10"`. Turing 0.29 → 0.46,
  DynamicPPL 0.23 → 0.42, AdvancedHMC 0.5 → 0.8, Pathfinder 0.8 → 0.10,
  MCMCChains 6 → 7, Optim 1 → 2. Turing is capped at 0.46 because Pathfinder
  0.10.7, the latest release, caps its Turing extension there. Widen the bound
  when Pathfinder does.
- **Removed dependencies:** AbstractDifferentiation, Accessors, AdvancedMH,
  Debugger, DelimitedFiles, Documenter (moved to `docs/Project.toml`),
  Formatting, LoggingExtras, Optimization, OptimizationOptimJL, ProgressMeter,
  ReverseDiff, SciMLBase, Serialization, SpecialFunctions.
- **`fit_cyclic_model(graph; kwargs...)` now runs the new pipeline** and
  returns `(fit, model, total_effects)`. The positional
  `fit_cyclic_model(g, log_normalize, model_pars, sampling_pars)` still works
  but warns and maps the old parameters onto the new ones. Use
  `fit_cyclic_model_legacy` for the published model.
- **`fit_model` and the discrete-DAG sampler are removed.** They were
  not part of the published analysis and were not included in the v0.2.0b
  module build; the source remains available at that tag.
- **`TotalEffectEstimate` has a new field, `knockdown_se`.** A 9-argument
  constructor without it is kept for code that builds the struct directly.
- **Dispersion is fixed by default.** `llcb_model(...; estimate_dispersion)`
  now defaults to `nothing`, meaning "sample `φ` only if there are residual
  degrees of freedom". With every readout perturbed there are none, so `φ ≡ 1`.
  Passing `true` without residual degrees of freedom warns.

### New

- `closed_form_coefficients(te)`: exact inversion of the total-effect
  identity, `B = I − T̃⁻¹ Diagonal(diag(T̃⁻¹))⁻¹`. This is the flat-prior MAP and
  costs one `p × p` inversion (12 ms at 300 genes, 0.3 s at 2,000). `has_full_total_effects(te)`
  reports whether it applies.
- `fit_llcb_closed_form(te; ndraws)`: uncertainty by resampling stage-1 total
  effects through the closed form, with no MCMC. It is unweighted, so weak
  knockdowns can distort it badly (see below).
- `weak_knockdowns(te; threshold = √10)`: rows whose first-stage F statistic
  is below 10. `estimate_total_effects` warns when any exist.
- `fit_llcb` now starts L-BFGS from the closed-form point when one exists
  (`diagnostics[:map_start]`). It pins BLAS to one thread while chains sample
  in parallel (`blas_threads = :auto`) and restores the setting afterwards.
- `fit_cyclic_model_legacy` and `joint_cyclic_model` are ported to current
  Turing and DynamicPPL (see below).

### Fixes

- **Data race in threaded stage 1** (introduced in the 0.3 development
  branch, never released). Threads wrote to shared `BitArray`s, which pack 64
  entries per word, so concurrent writes lost bits. A threaded run could
  silently drop a knockout row. These are now `Bool` arrays. In a regression
  test, 0 of 200 threaded runs differed from the serial run; a `BitMatrix`
  control lost bits in 190 of 200.
- **Dispersion funnel.** Sampling the unidentified `φ` produced a funnel. On a
  5-gene test, fixing it took `rhat` from 2.18 to 1.006 with 0 divergences.

### The published configuration in `fit_cyclic_model_legacy`

The port matches the settings the published analysis ran with:

1. Each Pathfinder run is initialised from Uniform(−0.03, 0.03), as in
   v0.2.0b (checked against Pathfinder 0.8.0 and 0.10.7). `init = :published`
   instead starts from the moment-based point computed in `fit_cyclic_model`
   (0.3·T\t, −3 for log σ, plus jitter).
2. The optimizer is `LBFGS(m = 8)`, as used by `wrap_multipathfinder`.
3. The noise term is written `MvNormal(μ, Diagonal(exp.(σ).^2))`, the current
   form of the original `MvNormal(μ, σvec)`, so `σₓ` remains a log standard
   deviation.

### Measured behaviour (simulation)

- **Head-to-head** (`bench/headtohead.jl`: 6 datasets, 8 genes, donor
  confounding). Published (ported): mean AP 0.460, RMSE 0.292, 95% coverage
  0.610. `fit_llcb`: AP 0.948, RMSE 0.122, coverage 0.958. Closed form
  (applicable to 5 of the 6 datasets): AP 0.952, RMSE 0.225, coverage 0.986.
- **Stability** (`bench/stability.jl`: 6 datasets, 10 genes). All six
  reached `rhat < 1.05` (max 1.015), with minimum ESS 308, coverage 0.963 and
  mean AP 0.886. The 0.3 development build reached `rhat < 1.05` on 4 of 6.
- **Stage 1** (`bench/stage1_study.jl`: 100 datasets per cell). The
  donor-adjusted Wald ratio beat the pooled slope in 66–88% of datasets, with
  27–58% lower median RMSE. Its mean RMSE was close to the pooled slope's, and
  higher in one cell, because weak knockdowns (F < 10, 6–13% of rows) give
  heavy tails. Excluding those rows, it won in 96–100% of datasets.
- **Scaling** (`bench/scaling.jl`, Julia 1.13). One log-density-plus-gradient
  evaluation takes 1.7 ms at 100 genes, 12.7 ms at 300 and 57 ms at 500 with a
  single BLAS thread. A single BLAS thread is fastest up to 300 genes (22.1 ms
  with eight threads at 300) and about 25% slower at 500, which supports
  pinning BLAS to one thread while chains sample in parallel.
- **Closed form with weak knockdowns.** In 10-gene simulations, RMSE was 0.082
  with no weak knockdowns and 1.1–5.3 with two. `fit_llcb` down-weights those
  rows by their standard errors and stayed at 0.08–0.16.

### Infrastructure

- CI tests Julia 1.10 and 1 on Linux, macOS and Windows with current
  `julia-actions`, runs `bench/validate_core.jl`, and adds a slow end-to-end
  job on pushes. The `validate_core.jl` stage-1 check now tests "wins in most
  datasets" and "wins on identified knockdowns" rather than the all-entries
  mean, which flipped between Julia versions' random streams.
- Docs build with Documenter 1 from `docs/Project.toml`.
- `Manifest.toml` is no longer committed, so `Pkg` resolves against the
  `[compat]` bounds on the user's Julia version. The v0.2.0b Manifest stays at
  that tag.
