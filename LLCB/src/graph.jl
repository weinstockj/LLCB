struct interventionGraph
    g::MetaDiGraph
    nv::Int64
    data::DataFrame
    x::Array{Float64}
    donor_map::Dict{String, Int8}
    donor::Array{Int8}
    interventions::Array{Int8}
end

"""


```julia-repl

using DataFrames

```
"""
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

function subset_to_interventions(data::DataFrame, metadata::DataFrame, intervention::String)
    # vnames = names(data)[contains.(names(data), intervention)]
    vnames = metadata.Sample[metadata.KO .== intervention]
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

function linear_regress(x, y::Vector{Float64})
    n = size(x, 1)
    @assert n == size(y, 1)
    @assert size(x, 2) >= 1
    @assert size(y, 2) == 1

    x = hcat(ones(n), x)

    βhat = x'x \ x'y

    return βhat[2]
end

function get_control_observations(g::interventionGraph)
    control_samples = vec(sum(g.interventions, dims = 2) .== 0) # indices where row sums == 0
    return control_samples
end

function get_experiment_observations(g::interventionGraph)
    
    targets = setdiff(names(g.data), ["donor", "intervention"])
    nv = length(targets)
    n  = size(g.x, 1)

    control_samples = get_control_observations(g)
    collection = Vector{BitArray}(undef, nv)
    for i in 1:nv
        experiment_observations = BitArray(g.interventions[:, i] + control_samples)
        collection[i] = deepcopy(experiment_observations)
    end

    return collection
end

function center_control_means(g::interventionGraph, x::Matrix{Float64})
    control_samples = get_control_observations(g)
    means = mean(x[control_samples, :]; dims = 1)
    return x .- means
end

function scale_control_sds(g::interventionGraph, x::Matrix{Float64})
    control_samples = get_control_observations(g)
    sds = std(x[control_samples, :]; dims = 1)
    return x ./ sds
end

function estimate_total_effects(g::interventionGraph, log_normalize::Bool, center=true, scale=true)
    targets = setdiff(names(g.data), ["donor", "intervention"])
    experiment_observations = get_experiment_observations(g)
    
    nv = length(targets)
    ndonors = length(unique(g.donor))

    # rows are parents, columns readouts
    total_effects = zeros(nv, nv)

    x = deepcopy(g.x)

    if log_normalize
        x = normalize(x)
    end

    if center
        x = center_control_means(g, x)
    end
    if scale
        x = scale_control_sds(g, x)
    end
   
    for i in 1:nv
        for j in 1:nv
            if i == j
                continue
            else
                # println("parent = $(targets[i]), child = $(targets[j])") 
                total_effects[i, j] = linear_regress(
                    x[experiment_observations[i], i], 
                    x[experiment_observations[i], j]
                )
            end
        end
    end
    
    return total_effects
end

function get_cyclic_matrices(g::interventionGraph, log_normalize::Bool, center=true, scale=true)

    total_effects = estimate_total_effects(g, log_normalize, center, scale)

    T = zeros(g.nv * (g.nv - 1), g.nv * (g.nv - 1))
    t = zeros(g.nv * (g.nv - 1))
    row = 0
    col = 0

    edges = Vector{Pair}()

    @inbounds for observed in 1:g.nv
        experiment_start = (observed - 1) * (g.nv - 1) + 1
        experiment_end = experiment_start + (g.nv - 1) - 1
        experiment_idx = experiment_start:experiment_end
        @assert length(experiment_idx) == g.nv - 1

        Tᵥ = total_effects[1:g.nv .!= observed, 1:g.nv .!= observed]
        Tᵥ[diagind(Tᵥ)] .= 1.0
        T[experiment_idx, experiment_idx] = Tᵥ
        t[experiment_idx] = total_effects[1:g.nv .!= observed, observed]

        @inbounds for e in 1:g.nv
            if e != observed
                push!(edges, Pair(e, observed))
            end
        end
    end

    return T, t, edges
end
