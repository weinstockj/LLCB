# Inside make.jl
push!(LOAD_PATH,"../src/")
using InferCausalGraph
using Documenter
makedocs(
         sitename = "InferCausalGraph.jl",
         modules  = [InferCausalGraph],
         pages=[
                "Home" => "index.md"
               ])
deploydocs(;
    repo="github.com/weinstockj/LLCB",
    devbranch = "dev"
)
