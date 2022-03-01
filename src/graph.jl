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
