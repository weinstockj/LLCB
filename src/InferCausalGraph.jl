module InferCausalGraph

using LinearAlgebra
using Statistics
using Random
using Dates
using SparseArrays

using CSV
using DataFrames
using Distributions
using Graphs
using MetaGraphs
using Memoize

using ADTypes
using AdvancedHMC
using DynamicPPL
using ForwardDiff
using LineSearches
using LogDensityProblems
using MCMCChains
using MCMCDiagnosticTools
using Optim
using Pathfinder
using Turing

include("total_effects.jl")      # stage 1 core: array level, stdlib only
include("graph.jl")              # interventionGraph + stage-1 adapters
include("data.jl")
include("expm.jl")
include("moment_model.jl")       # stage 2: exact moment likelihood, gradient, closed form; stdlib only
include("fit_llcb.jl")           # MAP / Pathfinder / NUTS, closed-form resampling
include("turing_models.jl")      # published Turing models, ported
include("legacy_cyclic.jl")      # published stage 2, ported
include("infer_cyclic_edges.jl")
include("infer_edges.jl")
include("utils.jl")

export interventionGraph,
       # stage 1
       estimate_total_effects, estimate_total_effects_legacy, TotalEffectEstimate,
       perturbed_nodes, weak_knockdowns,
       # stage 2
       llcb_model, LLCBModel, implied_total_effects, coefficients,
       closed_form_coefficients, has_full_total_effects,
       to_matrix, from_matrix, free_indices, initial_parameters, map_estimate,
       # fitting and summaries
       fit_llcb, fit_llcb_closed_form, LLCBFit, fit_cyclic_model, parse_llcb_fit,
       # published method (ported)
       fit_cyclic_model_legacy, joint_cyclic_model, parse_cyclic_chain,
       get_cyclic_matrices, get_model_params, get_sampling_params, spectral_radius

end
