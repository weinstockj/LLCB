struct DAGSampler <: Sampleable{Matrixvariate, Continuous}
    W::Matrix{AbstractFloat} # initial DAG
    dist::ContinuousDistribution # distribution of noise
end

struct DAG <: ContinuousMatrixDistribution
    W::Matrix{AbstractFloat} # initial DAG
    dist::ContinuousDistribution # distribution of noise
end

function DAGSampler(W::Matrix, μ::Float64, σ::Float64)
    A = (size(W, 1) != size(W, 2) ? deepcopy(convert_to_full_adjacency(W)) : deepcopy(W))

    size(A, 1) == size(A, 2) || error("matrix A is not square")

    sr = real(eigvals(A .* A)[size(A, 1)])
    if sr > .005
        error("W must be a DAG; instead, input DAG has spectral radius of $sr")
    end
    return DAGSampler(A, Normal(μ, σ))
end

function sampler(d::DAG)
    sampler = DAGSampler(d.W, d.dist)
    return sampler
end

function DAG(W::Matrix, μ::Float64, σ::Float64)
    A = (size(W, 1) != size(W, 2) ? deepcopy(convert_to_full_adjacency(W)) : deepcopy(W))

    size(A, 1) == size(A, 2) || error("matrix A is not square")

    sr = last(eigvals(A .* A))
    if sr > .005
        error("W must be a DAG; instead, input DAG has spectral radius of $sr")
    end
    return DAG(A, Normal(μ, σ))
end

# Base.size(s::DAGSampler) = size(convert_to_reduced_adjacency(s.W))
# Base.size(d::DAG) = size(convert_to_reduced_adjacency(d.W))
Base.size(s::DAGSampler) = size(s.W)
Base.size(d::DAG) = size(d.W)

function reverse(x::CartesianIndex)
    return CartesianIndex(last(Tuple(x)), first(Tuple(x)))
end

function _rand!(rng::AbstractRNG, s::DAGSampler, x::Matrix{T}) where T<:AbstractFloat

    # println("debug me 2")
    # println("size(s.W) = $(size(s.W))")
    # println("size(x) = $(size(x))")
    d = size(s.W, 2) # genes 
    @assert d == size(s.W, 1)

    dist = s.dist

    max = 30
    count = 0
    
    non_zero_edges = findall(abs.(x) .> 0.0)
    reversed_edges = reverse.(non_zero_edges)
    zero_edges = findall(x .== 0.0)
    legal_new_edges = setdiff(zero_edges, reversed_edges)
    choices = ("delete", "reverse", "add", "modify")
    choice_dist = Categorical([.25, .25, .25, .25])
    local edge
    local reverse_edge

    while count < max
        count += 1
        proposal_choice = choices[rand(choice_dist)]
        # println("choice = $proposal_choice")
        if proposal_choice == "delete"
            edge = sample(non_zero_edges)
            x[edge] = 0
            # i = Base.rand(1:d)
            # j = Base.rand(setdiff(1:d, i)) # no diagonals

        elseif proposal_choice == "reverse"
            edge = sample(non_zero_edges)
            # reverse_edge = CartesianIndex(last(Tuple(edge)), first(Tuple(edge)))
            reverse_edge = reverse(edge)
            x[reverse_edge] = x[edge] 
            x[edge] = 0.0
        elseif proposal_choice == "modify"
            edge = sample(non_zero_edges)
            x[edge] += rand(dist)
        else # add new edge
            edge = sample(legal_new_edges)
            x[edge] = rand(dist)
        end
        sr = last(eigvals(x .* x))
        # println("sr = $sr")
        if (typeof(sr) <: Real) && (sr < 0.005)
            # x .= x .- s.W # for random walk proposal
            # println("found a DAG; proposal = $proposal_choice, x = $x, count = $count, sr = $sr")
            return
        else
            # println("not a DAG; proposal = $proposal_choice,  x = $x, count = $count, sr = $sr")
            x[edge] = s.W[edge] # refer to original entry
            if @isdefined reverse_edge
                x[reverse_edge] = s.W[reverse_edge] # refer to original entry
            end
        end
    end

    error("can't sample new DAG after $max attempts")

end

function Base.rand(rng::AbstractRNG, s::DAGSampler)
    # x = (size(s.W, 1) != size(s.W, 2) ? deepcopy(convert_to_full_adjacency(s.W)) : deepcopy(s.W))
    x = deepcopy(s.W)

    _rand!(rng, s, x)

    # println("x=$x")
    # return convert_to_reduced_adjacency(x)
    return x
end

function Base.rand(d::DAG)
    return Base.rand(sampler(d))
end

function Base.rand(rng::AbstractRNG, d::DAG)
    return Base.rand(sampler(d))
end


function Distributions._logpdf(d::DAG, X::AbstractMatrix{T}) where T<:AbstractFloat
    # println("debug me 3")
    # use the below for a static proposal
    γ = 0.2
    val = exp(-γ * norm(X, 1))
    return val
    # return 0.0
end

