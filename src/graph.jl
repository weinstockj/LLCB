struct interventionGraph
    g::MetaDiGraph
    nv::Int64
    data::DataFrame
    x::Array{Float64}
    donor_map::Dict{String, Int8}
    donor::Array{Int8}
    interventions::Array{Int8}
end

# interventionGraph = interventionGraph3

function interventionGraph(data::DataFrame)
    default_weight = 0.0

    targets = setdiff(names(data), ["donor", "intervention"])
    nv = length(targets)
    @info "$(now()) intervention graph has $nv nodes"
    nrows = nrow(data)

    x = Matrix(@view data[:, targets])
    interventions = Int8.(zeros(nrows, nv))

    unique_donors = unique(data.donor)
    donor_map = Dict{String, Int8}()
    donor = Array{Int8}(undef, nrows)

    for i in 1:length(unique_donors)
        donor_map[unique_donors[i]] = Int8(i)
    end

    for i in 1:nrows
        donor[i] = donor_map[data.donor[i]]
        for j in 1:nv
            if data[i, :intervention] == targets[j]
                interventions[i, j] = 1
                @assert x[i, j] == 0.0
            end
        end
    end

    graph = interventionGraph(
        MetaDiGraph(SimpleDiGraph(nv), default_weight),
        nv,
        data,
        x,
        donor_map,
        donor,
        interventions
    )

    @info "$(now()) done constructing graph"
    set_indexing_prop!(graph.g, :name)

    return graph
end

function subset_to_interventions(data::DataFrame, intervention::String)
    vnames = names(data)[contains.(names(data), intervention)]
    push!(vnames, "readout_gene")
    return data[:, vnames]
end

function posterior_graph(parsed_chain::DataFrame, beta_threshold::Float64 = 0.02)

    filtered_chain = @view parsed_chain[abs.(parsed_chain.estimate) .>= beta_threshold, :]
    n_rows = nrow(filtered_chain)
    nodes = unique(vcat(filtered_chain.row, filtered_chain.col))
    n_vertices = length(nodes)
    n_edges = n_rows # including for clarity..
    @info "$(now()) Now constructing graph from $n_vertices nodes and $n_edges edges"

    default_weight = 0.0
    g = MetaDiGraph(SimpleDiGraph(n_vertices), default_weight)

    for i in 1:n_rows
        x_node_index = findall(x -> x == filtered_chain.row[i], nodes)[1]
        y_node_index = findall(x -> x == filtered_chain.col[i], nodes)[1]
        add_edge!(g, x_node_index, y_node_index)
        set_prop!(g, Edge(x_node_index, y_node_index), :weight, filtered_chain.estimate[i])
    end

    for i in 1:n_vertices
        set_prop!(g, i, :name, nodes[i])
    end

    return g
end

function posteriorGraph!(graph::interventionGraph, parsed_chain::DataFrame, beta_threshold::Float64 = 0.02)
    filtered_chain = @view parsed_chain[abs.(parsed_chain.estimate) .>= beta_threshold, :]
    n_rows = nrow(filtered_chain)
    nodes = unique(vcat(filtered_chain.row, filtered_chain.col))
    n_vertices = length(nodes)
    n_edges = n_rows # including for clarity..
    @info "$(now()) Now constructing graph from $n_vertices nodes and $n_edges edges"

    for i in 1:n_rows
        x_node_index = findall(x -> x == filtered_chain.row[i], nodes)[1]
        y_node_index = findall(x -> x == filtered_chain.col[i], nodes)[1]
        add_edge!(graph.g, x_node_index, y_node_index)
        set_prop!(graph.g, Edge(x_node_index, y_node_index), :weight, filtered_chain.estimate[i])
    end

    for i in 1:n_vertices
        set_prop!(graph.g, i, :name, nodes[i])
    end

    for i in (n_vertices + 1):nv(graph.g)
        # TODO: need to run this multiple times??
        rem_vertex!(graph.g, i) || error("failed to remove node $i") # remove superflous nodes
    end

end

function topological_sort(G::BitMatrix)
     
    g = deepcopy(G)
    L = Vector{Int}()
    S = Vector{Int}() # nodes with no parents
    nv = size(G, 1)
    @assert nv == size(G, 2)

    for l in 1:nv
        if all(sum(g[:, l]) == 0)
            push!(S, l)
        end
    end

    while length(S) > 0
        n = popfirst!(S)
        push!(L, n)
        children = findall(x -> x != 0, g[n, :])
        for m in children
            g[n, m] = 0
            parents = findall(x -> x != 0, g[:, m])
            if length(parents) == 0
                push!(S, m)
            end
        end
    end

    if sum(g) != 0
        # error("graph has a cycle")
        return nothing
    else
        return L
    end
end


function gauss_ci_test(n::Int64, C::Matrix{Float64}, i, j, S = (), α = 0.05)

    if length(S) > 0
        @assert length(intersect(vcat(i, j), S)) == 0
    end

    if length(S) == 0
        r = C[i, j]
    elseif length(S) == 1
        k = S[1]
        r = (C[i, j] - C[i, k] * C[j, k]) / sqrt((1 - C[j, k] ^ 2) * (1 - C[i, k] ^ 2))
    else 
        θ = inv(C[vcat(i, j, S), vcat(i, j, S)])
        r = -θ[1, 2] / sqrt(θ[1, 1] * θ[2, 2])
    end

    critical = quantile(Normal(), 1 - α / 2.0)
    stat = sqrt(n - length(S) - 3) * abs(.5 * log1p(2 * r / (1.0 - r)))
    return abs(stat) > critical # return true if reject
end

# mutable struct SufficientStats
#     n::Int64
#     μhat::Vector{Float64}
#     Σhat::Matrix{Float64}
# end
#
# function SufficientStats(X::Matrix{Float64})
#     n = size(X, 1)
#     μhat = vec(Statistics.mean(X, dims=1))
#     Σhat = cor(X)
#
#     return SufficientStats(
#         n,
#         μhat,
#         Σhat
#     )
# end

function IMAP(perm::Vector{Int64}, α, X::Matrix{Float64}, interventions)
    
    nv = length(perm)
    N = size(X, 1)

    G = BitMatrix(zeros(nv, nv))

    for i in 1:(nv - 1)
        for j in (i+1):nv
            parents = 1:(j-1) # indices in perm
            S = [perm[p] for p in parents] # node ids of parents
            S = setdiff(S, perm[i])
            # observed = interventions[:, j] .!= 1
            observed = 1:N
            XO = @view X[observed, :]
            ci_test = gauss_ci_test(size(XO, 1), cor(XO), perm[i], perm[j], S, α)
            if ci_test
                G[perm[i], perm[j]] = 1
            end
        end
    end

    return G
end
