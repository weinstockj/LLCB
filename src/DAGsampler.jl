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
    T0::Matrix{AbstractFloat} # prior for Wishart
    aᵤ::AbstractFloat
    aw::AbstractFloat
    nv::Integer # nodes in graph
end

function DAGScorer(G::BitMatrix)

    nv = size(G, 1)
    ESS = nv + 20
    aᵤ = 1.0
    aw = nv + aᵤ + 1.0
    T0 = aᵤ * (aw - nv - 1) / (aᵤ + 1) * Matrix(I, nv, nv)
    @assert size(T0, 1) == size(T0, 2)

    return DAGScorer(
        G,
        ESS,
        T0,
        aᵤ,
        aw,
        nv
    )
end


# compute bge score
function bge(d::DAGScorer, X::AbstractMatrix{T}) where T<:AbstractFloat


    N = size(X, 1) 
    μ0 = zeros(d.nv)
    μhat = Statistics.mean(X; dims = 1) # row vector
    XC = X .- μhat
    μhat = vec(μhat)
    @assert size(XC) == size(X)
    Σhat = cov(X)
    # println("μhat = $μhat")
    # println("N = $N")

    @assert length(μhat) == d.nv
    @assert size(Σhat)[1] == d.nv
    @assert size(Σhat)[2] == d.nv

    TN = d.T0 .+ (Σhat .* (N - 1)) .+ (N * d.aᵤ / (N + d.aᵤ)) .* (μ0 - μhat) * (μ0 - μhat)' 

    # println("Σhat = $Σhat")
    # println("R = $R")

    @assert size(TN, 1) == size(TN, 2)
    @assert size(TN, 1) == d.nv
    awpN = d.aw + N

    μN = (N * μhat + d.aᵤ * μ0) / (N + d.aᵤ)
    ΣN = Hermitian(TN / (awpN - d.nv - 1))

    # println("μN = $μN")
    # println("ΣN ")
    # Base.print_matrix(stdout, round.(ΣN, digits = 3))
    #
    # const_term = (-N / 2.0) * log(π) + 0.5 * log(d.aᵤ / (N + d.aᵤ))
    # loglik = 0.0
    #
    # const_vec = zeros(d.nv)
    T0scale = d.T0[1, 1]

    # for j in 1:d.nv # number of parents + 1
    #     awp = d.aw - d.nv + j
    #     const_vec[j] = const_term - lgamma(awp / 2.0) + lgamma((awp + N) / 2) + ((awp + j - 1) / 2) * log(T0scale)
    #
    # end

    scores = zeros(N, d.nv)

    for i in 1:d.nv
        A = ΣN[i, i]
        parents = findall(d.G[:, i] .!= 0)
        n_parents = length(parents)
        # println("node $i has $n_parents parents")
        if n_parents == 0
           scores[:, i] .= (-XC[:, i] .^ 2) ./ (2 * A) .- log(2.0 * π * A) / 2.0 
        else
            D = ΣN[parents, parents]
            # Base.print_matrix(stdout, round.(D, digits = 3))
            chol = cholesky(D)
            B = ΣN[i, parents]
            C = chol.L \ B
            E = chol.U \ C
            # println()
            # println("C")
            # println()
            # Base.print_matrix(stdout, round.(C, digits = 3))
            # println()
            # println("E")
            # println()
            # Base.print_matrix(stdout, round.(E, digits = 3))

            K = A - sum(C .^ 2)
            # println("K = $K")
            @assert size(K) == ()
            @assert K > 0

            coreMat = vcat(1, -1.0 .* E) * (vcat(1, -1.0 .* E))' ./ K
            xs = @view XC[:, vcat(i, parents)]

            # println("coreMat")
            # Base.print_matrix(stdout, round.(coreMat, digits = 3))
            tmp  = (xs * coreMat) .* xs
            scores[:, i] .= -sum(tmp, dims=2) ./ 2.0 .- log(2 * π * K) / 2
        end
    end

    # for i in 1:d.nv
    #     node_loglik = 0.0
    #     parents = findall(d.G[:, i] .!= 0)
    #     family = union([i], parents) # child node and parents
    #
    #     println("parents for node $i are $parents")
    #
    #     n_parents = length(parents)
    #     
    #
    #     for j in 0:Int(floor(N / 2.0))
    #         node_loglik += log(0.5 * (2 * i + d.ESS - d.nv + n_parents + 1))
    #     end
    #     node_loglik += (0.5 * (d.ESS - d.nv + 2 * n_parents + 1)) * log(d.aᵤ * (d.ESS - d.nv - 1) / (d.aᵤ + 1.0))
    #     node_loglik -= (0.5 * (N + d.ESS - d.nv + n_parents + 1)) * log(det(R[family, family]))
    #
    #     if n_parents > 0
    #         node_loglik += (0.5 * (N + d.ESS - d.nv + n_parents)) * log(det(R[parents, parents])) 
    #     end
    #     loglik += node_loglik
    # end

    return sum(scores)
end

Base.size(d::DAG) = (d.nv, d.nv)

function reverse(x::CartesianIndex)
    return CartesianIndex(last(Tuple(x)), first(Tuple(x)))
end

function _rand!(rng::AbstractRNG, d::DAG, x::Matrix{Float64}) 

    # println("debug1 me here!, x=$x")
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

    # println("debug2 me here!")
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
        
        # println("debug3 me here!")
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
    # x = BitMatrix(zeros(d.nv, d.nv))
    x = zeros(d.nv, d.nv)

    _rand!(rng, d, x)

    return x
end

function Distributions.logkernel(d::DAG, X::BitMatrix)
    return 0.0
end

# function Distributions.logpdf(d::DAG, X::BitMatrix) 
#     return Distributions._logpdf(d, X)
# end
#

function Distributions._logpdf(d::DAG, X::Matrix{Float64})
    # println("size(X) = $(size(X))")
    # println("X = $X")
    return Distributions._logpdf(d, BitMatrix(X))
end

function Distributions._logpdf(d::DAG, X::BitMatrix)
    # println("debug me 3")
    # use the below for a static proposal
    γ = 3.5
    val = exp(-γ * norm(X, 1))
    return val
    # return 0.0
end

