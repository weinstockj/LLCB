# struct DAGSampler <: Sampleable{Matrixvariate, Continuous}
#     nv::Integer
# end

struct DAG <: ContinuousMatrixDistribution
    nv::Integer
    logc0::AbstractFloat
end

function DAG(nv::Integer)
    return DAG(nv, 0.0)
end

struct DAGScorer
    G::BitMatrix # unweighted adjacency matrix
    ESS::Integer
    T::Matrix{AbstractFloat} # prior for Wishart
    aᵤ::AbstractFloat
    nv::Integer # nodes in graph
end

function DAGScorer(G::BitMatrix)

    nv = size(G, 1)
    ESS = nv + 20
    aᵤ = 1.0
    T = aᵤ * (ESS - nv - 1) / (aᵤ + 1) * Matrix(I, nv, nv)
    @assert size(T, 1) == size(T, 2)

    return DAGScorer(
        G,
        ESS,
        T,
        aᵤ,
        nv
    )
end


function ll(d::DAGScorer, X::AbstractMatrix{T}) where T<:AbstractFloat

    N = size(X, 1) 
    μhat = Statistics.mean(X; dims = 1)
    Σhat = cov(X)
    # println("μhat = $μhat")
    # println("N = $N")

    @assert length(μhat) == d.nv
    @assert size(Σhat)[1] == d.nv
    @assert size(Σhat)[2] == d.nv

    R = d.T .+ Σhat .+ (N * d.ESS / (N + d.ESS)) .* sum(μhat .* μhat)

    # println("Σhat = $Σhat")
    # println("R = $R")

    @assert size(R, 1) == size(R, 2)
    @assert size(R, 1) == d.nv

    loglik = 0.0

    for i in 1:d.nv
        node_loglik = 0.0
        parents = findall(d.G[:, i] .!= 0)
        family = union([i], parents) # child node and parents

        println("parents for node $i are $parents")

        n_parents = length(parents)
        
        node_loglik = (-N / 2.0) * log(π) + 0.5 * log(d.aᵤ / (N + d.aᵤ))

        for j in 0:Int(floor(N / 2.0))
            node_loglik += log(0.5 * (2 * i + d.ESS - d.nv + n_parents + 1))
        end
        node_loglik += (0.5 * (d.ESS - d.nv + 2 * n_parents + 1)) * log(d.aᵤ * (d.ESS - d.nv - 1) / (d.aᵤ + 1.0))
        node_loglik -= (0.5 * (N + d.ESS - d.nv + n_parents + 1)) * log(det(R[family, family]))

        if n_parents > 0
            node_loglik += (0.5 * (N + d.ESS - d.nv + n_parents)) * log(det(R[parents, parents])) 
        end
        loglik += node_loglik
    end

    return loglik
end

Base.size(d::DAG) = (d.nv, d.nv)

function reverse(x::CartesianIndex)
    return CartesianIndex(last(Tuple(x)), first(Tuple(x)))
end

function _rand!(rng::AbstractRNG, d::DAG, x::BitMatrix) 

    println("debug me here!")
    p = d.nv # genes 

    max = 30
    count = 0

    orig = deepcopy(x)
    
    non_zero_edges = findall(abs.(x) .> 0.0)
    reversed_edges = reverse.(non_zero_edges)
    zero_edges = findall(x .== 0.0)
    legal_new_edges = setdiff(zero_edges, reversed_edges)
    choices = ("delete", "reverse", "add")
    choice_dist = Categorical([0, 0, 1.0])
    local edge
    local reverse_edge

    while count < max
        count += 1
        proposal_choice = choices[rand(choice_dist)]
        if proposal_choice == "delete"
            edge = sample(non_zero_edges)
            x[edge] = 0
        elseif proposal_choice == "reverse"
            edge = sample(non_zero_edges)
            reverse_edge = reverse(edge)
            x[reverse_edge] = x[edge] 
            x[edge] = 0.0
        else # add new edge
            edge = sample(legal_new_edges)
            x[edge] = 1.0
        end
        
        sr = last(eigvals(x .* x))
        # println("sr = $sr")
        if (typeof(sr) <: Real) && (sr < 0.005)
            # println("found a DAG; proposal = $proposal_choice, x = $x, count = $count, sr = $sr")
            return
        else
            # println("not a DAG; proposal = $proposal_choice,  x = $x, count = $count, sr = $sr")
            x[edge] = orig[edge] # refer to original entry
            if @isdefined reverse_edge
                x[reverse_edge] = orig[reverse_edge] # refer to original entry
            end
        end
    end

    error("can't sample new DAG after $max attempts")

end


function Base.rand(rng::AbstractRNG, d::DAG)
    x = BitMatrix(zeros(d.nv, d.nv))

    _rand!(rng, d, x)

    return x
end

function Distributions.logkernel(d::DAG, X::BitMatrix)
    return 0.0
end

# function Distributions.logpdf(d::DAG, X::BitMatrix) 
#     return Distributions._logpdf(d, X)
# end

function Distributions._logpdf(d::DAG, X::BitMatrix)
    # println("debug me 3")
    # use the below for a static proposal
    γ = 0.2
    val = exp(-γ * norm(X, 1))
    return val
    # return 0.0
end

