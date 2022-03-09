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

Base.size(s::DAGSampler) = size(convert_to_reduced_adjacency(s.W))
Base.size(d::DAG) = size(convert_to_reduced_adjacency(d.W))

function _rand!(rng::AbstractRNG, s::DAGSampler, x::Matrix{T}) where T<:AbstractFloat

    # println("debug me 2")
    # println("size(s.W) = $(size(s.W))")
    # println("size(x) = $(size(x))")
    d = size(s.W, 2) # genes 
    @assert d == size(s.W, 1)

    dist = s.dist

    max = 20
    count = 0

    while count < max
        count += 1
        i = Base.rand(1:d)
        j = Base.rand(setdiff(1:d, i)) # no diagonals
        tmp = rand(dist)
        x[i, j] += tmp
        sr = last(eigvals(x .* x))
        # println("sr = $sr")
        if (typeof(sr) <: Real) && (sr < 0.005)
            # println("x = $x, count = $count")
            return
        else
            x[i, j] = s.W[i, j] # refer to original entry
        end
    end

    error("can't sample new DAG after $max attempts")

end

function _rand!(rng::AbstractRNG, d::DAG, x::Matrix{T}) where T<:AbstractFloat
    g = size(d.W, 2) # genes 
    @assert g == size(d.W, 1)

    dist = s.dist

    max = 20
    count = 0

    while count < max
        count += 1
        i = Base.rand(1:g)
        j = Base.rand(setdiff(1:g, i)) # no diagonals
        tmp = rand(dist)
        x[i, j] += tmp
        sr = real(eigvals(x .* x)[g])
        # println("x = $x")
        # println("sr = $sr")
        if (typeof(sr) <: Real) && (sr < 0.005)
            return
        else
            x[i, j] = s.W[i, j] # refer to original entry
        end
    end

    @assert !isequal(x, d.W) # don't return original DAG...

    error("can't sample new DAG after $max attempts")


end

function Base.rand(rng::AbstractRNG, s::DAGSampler)
    # x = (size(s.W, 1) != size(s.W, 2) ? deepcopy(convert_to_full_adjacency(s.W)) : deepcopy(s.W))
    x = deepcopy(s.W)

    _rand!(rng, s, x)

    # println("x=$x")
    return convert_to_reduced_adjacency(x)
end

function Base.rand(d::DAG)
    return Base.rand(sampler(d))
end

function Base.rand(rng::AbstractRNG, d::DAG)
    return Base.rand(sampler(d))
end


function Distributions._logpdf(d::DAG, X::AbstractMatrix{T}) where T<:AbstractFloat
    # println("debug me 3")
    return 0.0
end

function Distributions.logkernel(d::DAG, X::AbstractMatrix{T}) where T<:AbstractFloat
    println("debug me 4")
    return 0.0
end
