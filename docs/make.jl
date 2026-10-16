push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using InferCausalGraph
using Documenter

makedocs(;
    sitename = "InferCausalGraph.jl",
    modules = [InferCausalGraph],
    pages = ["Home" => "index.md", "Methods" => "methods.md"],
    # internal helpers are documented in-source but not all listed in the manual
    warnonly = [:missing_docs],
)
deploydocs(; repo = "github.com/weinstockj/LLCB", devbranch = "dev")
