using Distributions
using LinearAlgebra: I
using InferCausalGraph: d_connected_advi, parse_symbol_map
using Statistics: var
using Formatting: sprintf1


@testset "Skeleton estimation" begin
    include("extended_test.jl")

    # conduct 10 sims
    sims = [sim() for i in 1:10]

    println(sims)

    beta_x_error = mean([i.beta_x_error for i in sims])
    beta_x_interact_i = mean([i.beta_x_interact_i for i in sims])
    beta_pip = mean([i.beta_pip for i in sims])

    @test abs(beta_x_error) < 0.05 # edge weight close to true value
    @test abs(beta_x_interact_i) < 0.20 # should be small
    @test beta_pip > 0.50
    
end
