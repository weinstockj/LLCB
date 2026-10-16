
using InferCausalGraph
using Test
using LinearAlgebra
using Statistics
using Random
using DataFrames

# The fast tests exercise the numerical core directly. They do not touch the
# sampling stack, so they run in a few seconds.
include("core_tests.jl")

@testset "InferCausalGraph.jl" begin
    @testset "numerical core" begin
        test_moment_identity()
        test_analytic_gradient()
        test_closed_form()
        test_log_density_determinism()
        test_packing_roundtrip()
        test_total_effects_recovery()
        test_total_effects_beats_pooled_ols()
    end

    @testset "interventionGraph plumbing" begin
        test_intervention_graph_roundtrip()
    end

    @testset "legacy reference" begin
        test_linear_regress_legacy()
        test_spectral_radius_legacy()
    end

    # Full sampling runs are slow (minutes); enable with
    #   LLCB_RUN_SLOW_TESTS=1 julia --project -e 'using Pkg; Pkg.test()'
    if get(ENV, "LLCB_RUN_SLOW_TESTS", "0") == "1"
        @testset "end-to-end fit" begin
            test_end_to_end_fit()
        end
    else
        @info "skipping the end-to-end sampling test; set LLCB_RUN_SLOW_TESTS=1 to run it"
    end
end
