struct interventionGraph
    g::MetaDiGraph
    nv::Int64
    data::DataFrame
    hasIntervention::Bool
    interventions::Union{Missing, Vector{String}}
end

function interventionGraph(data::DataFrame, interventions::Vector{String})
    default_weight = 0.0
    nv = ncol(data) - 1
    @info "$(now()) intervention graph has $nv nodes"
    graph = interventionGraph(
        # MetaDiGraph(complete_graph(nv), default_weight),
        MetaDiGraph(SimpleGraph(nv), default_weight),
        nv,
        data,
        length(interventions) >= 1 ? true : false,
        interventions
    )
    colnames = names(data)

    for i in 1:nv
        # @info "$(now()) node is $(data.readout_gene[i])"
        # @info "$(now()) node is $(colnames[i])"
        #   
        col_index = i + 1
        set_prop!(graph.g, i, :name, colnames[col_index])
        if issubset(colnames[col_index], interventions)
            n = inneighbors(graph, i)
            for v in 1:n
               rem_edge!(graph.g, n, i) # remove all incomming edges to intervened nodes 
            end
        end
    end

    @info "$(now()) done constructing graph"
    set_indexing_prop!(graph.g, :name)

    return graph
end

function subset_to_interventions(data::DataFrame, intervention::String)
    vnames = names(data)[contains.(names(data), intervention)]
    push!(vnames, "readout_gene")
    return data[:, vnames]
end

struct interventionGraphSet
    n_graphs::Int64
    graphs::Array{interventionGraph}
    data::DataFrame
end

function interventionGraphSet(data_collection::Dict, interventions::Vector{String})
    n_interventions = length(interventions)
    graph_collection = interventionGraphSet(
        n_interventions,
        Array{interventionGraph}(undef, n_interventions),
        reduce(vcat, values(data_collection))
    )

    for i in range(1, stop = n_interventions)
        graph_collection.graphs[i] = interventionGraph(
            data_collection[interventions[i]], 
            [interventions[i]]
        )
    end

    return graph_collection
end
    
