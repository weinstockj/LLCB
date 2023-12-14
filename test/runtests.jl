using InferCausalGraph
using Test
using LinearAlgebra: eigen
using Dates: now

@testset "InferCausalGraph.jl" begin
    # Write your tests here.
    # include("test_skeleton.jl")
    include("extended_test.jl")
    
    function run_cylic_model_test()
        println("$(now()) Now running cyclic model test")
        m = sim_cyclic_expression_and_fit_model()
        parsed = m[3]
        parsed_threshold = parsed[parsed.PIP >= .95]
        println("$(now()) Done fitting model")

        parsed_threshold = run_test()

        detected_edges = Set(collect(zip(parsed_threshold.row, parsed_threshold.col)))

        true_edges = Set([
            ("gene_5", "gene_1"),
            ("gene_1", "gene_2"),
            ("gene_2", "gene_3"),
            ("gene_3", "gene_4"),
            ("gene_4", "gene_5"),
           ])
        
        intersect_edges = intersect(detected_edges, true_edges)
        false_positive_edges = setdiff(detected_edges, true_edges)
        @test length(intersect_edges) == 5
        @test length(false_positive_edges) <= 1
    end


    # run_cylic_model_test() # commented out because it's slow in unit testing env
    test_linear_regress()

    function test_spectral_radius()
        A = rand(10, 10)
        A = A + A'
        B = A .* A

        max_eigen_value = last(eigen(B).values)
        test_eigen_value = spectral_radius(A, 40, 1e-8)

        @test abs(max_eigen_value - test_eigen_value) / max_eigen_value < .01
    end

    test_spectral_radius()
end
