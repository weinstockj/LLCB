using InferCausalGraph
using Test
using Dates: now

@testset "InferCausalGraph.jl" begin
    # Write your tests here.
    # include("test_skeleton.jl")
    include("extended_test.jl")
    
    function run_test()
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


    # run_test()
    test_linear_regress()

end
