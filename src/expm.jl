using LinearAlgebra
import Base.exp
import LinearAlgebra.exp!

# some of the folowing code copied from: https://github.com/JuliaDiff/ForwardDiff.jl/issues/174#issuecomment-997443414
function expmchk()
# EXPMCHK Check the class of input A and
#    initialize M_VALS and THETA accordingly.
    m_vals = [3 5 7 9 13];
    # theta_m for m=1:13.
                theta = [#3.650024139523051e-008
                         #5.317232856892575e-004
                          1.495585217958292e-002  # m_vals = 3
                         #8.536352760102745e-002
                          2.539398330063230e-001  # m_vals = 5
                         #5.414660951208968e-001
               
                         9.504178996162932e-001  # m_vals = 7
                         #1.473163964234804e+000
                          2.097847961257068e+000  # m_vals = 9
                         #2.811644121620263e+000
                         #3.602330066265032e+000
                         #4.458935413036850e+000
                          5.371920351148152e+000];# m_vals = 13
    
    return m_vals, theta
end

function getPadeCoefficients(m)
# GETPADECOEFFICIENTS Coefficients of numerator P of Pade approximant
#    C = GETPADECOEFFICIENTS returns coefficients of numerator
#    of [M/M] Pade approximant, where M = 3,5,7,9,13.
    if (m==3)
        c = [120, 60, 12, 1];
    elseif (m==5)
        c = [30240, 15120, 3360, 420, 30, 1];
    elseif (m==7)
        c = [17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1];
    elseif (m==9)
        c = [17643225600, 8821612800, 2075673600, 302702400, 30270240, 
             2162160, 110880, 3960, 90, 1];
    elseif (m==13)
        c = [64764752532480000, 32382376266240000, 7771770303897600, 
             1187353796428800,  129060195264000,   10559470521600, 
             670442572800,      33522128640,       1323241920,
             40840800,          960960,            16380,  182,  1];
    end
        
    return c      
end

function PadeApproximantOfDegree(A,m)
#PADEAPPROXIMANTOFDEGREE  Pade approximant to exponential.
#   F = PADEAPPROXIMANTOFDEGREE(M) is the degree M diagonal
#   Pade approximant to EXP(A), where M = 3, 5, 7, 9 or 13.
#   Series are evaluated in decreasing order of powers, which is
#   in approx. increasing order of maximum norms of the terms.

    n = maximum(size(A));
    c = getPadeCoefficients(m);

    # Evaluate Pade approximant.
    if (m == 13)
        # For optimal evaluation need different formula for m >= 12.
        A2 = A*A
        A4 = A2*A2 
        A6 = A2*A4
        U = A * (A6*(c[14]*A6 + c[12]*A4 + c[10]*A2) + c[8]*A6 + c[6]*A4 + c[4]*A2 + c[2]*Matrix{Float64}(I,n,n) )
        V = A6*(c[13]*A6 + c[11]*A4 + c[9]*A2) + c[7]*A6 + c[5]*A4 + c[3]*A2 + c[1]*Matrix{Float64}(I,n,n)
                
        F = (V-U)\(V+U)

else # m == 3, 5, 7, 9
        Apowers = Array{Matrix{Float64}}(undef,ceil(Int,(m+1)/2))
        
        Apowers[1] = Matrix{Float64}(I,n,n)
        Apowers[2] = A*A
        
        for j = 3:ceil(Int,(m+1)/2)
            Apowers[j] = Apowers[j-1]*Apowers[2]
        end
    
        U = zeros(n,n) 
        V = zeros(n,n)
        
        for j = m+1:-2:2
            U = U + c[j]*Apowers[convert(Int,j/2)]
        end
    
        U = A*U
        
        for j = m:-2:1
            V = V + c[j]*Apowers[convert(Int,(j+1)/2)]
        end
    
        F = (V-U)\(V+U)
    end
    
    return F
end

function expm(A)
# EXPM   Matrix exponential.
#   EXPM(X) is the matrix exponential of X.  EXPM is computed using
#   a scaling and squaring algorithm with a Pade approximation.
#
# Julia implementation closely based on MATLAB code by Nicholas Higham
#
    # Initialization
    m_vals, theta = expmchk()

    normA = norm(A,1)

    if normA <= theta[end]
        # no scaling and squaring is required.
        for i = 1:length(m_vals)
            if normA <= theta[i]
                F = PadeApproximantOfDegree(A,m_vals[i])
                break
            end
        end
    else
        tmp = normA / theta[end]
        t = frexp1(tmp)
        s = frexp2(tmp)
        s = s - (t == 0.5) # adjust s if normA/theta(end) is a power of 2.
        A = A/2^s          # Scaling
        F = PadeApproximantOfDegree(A,m_vals[end])
        
        for i = 1:s
            F = F*F   # Squaring
        end
    end

    return F
end # expm

frexp1(x) = frexp(x)[1]
frexp2(x) = frexp(x)[2]
dfrexp1(x) = 2.0^(-floor(log2(abs(x))) - 1)
dfrexp2(x) = 0

for (f, df) in ((:frexp1, :dfrexp1), (:frexp2, :dfrexp2))
    @eval begin
        function ($f)(d::ForwardDiff.Dual{T}) where T

            x = ForwardDiff.value(d)
            y = ($f)(x)
            dy = ($df)(x)
            return ForwardDiff.Dual{T}(y, dy * ForwardDiff.partials(d))

        end
    end
end

# exp(x::Matrix{T}) where T<:Real = expm(x)
# function exp!(x::Matrix{T}) where T<:Real
#     x[:] = expm(x)
# end
#
#
"""
`spectral_radius(W::Matrix{AbstractFloat})`

Returns the largest eigenvalue of `W .* W` computed
through power iteration. This is used in the model
fitting process to penalize the adjacency matrix. 

"""
function spectral_radius(W, max_iter = 30, ϵ = 1e-7) 
    B = W .* W
    x₀ = Base.rand(size(B, 1))
    y = x₀
    λold = 0.0
    λ = 1.0
    for i in 1:max_iter
        y = B * y
        y = y / norm(y)
        λ = dot(y, B * y) / dot(y, y)
        if abs(λ - λold) < ϵ
            break
        end
        λold = λ
    end
    return λ
end

function notears(W::Matrix{T}) where T
    # println("notears is being called")
    # value = tr(exp(W .* W)) - size(W, 1)
    p = size(W, 1)
    # value = tr(padm(W .* W)) - p
    # value = tr(exp(W .* W)) - p
    # value = tr(expv(1.0, W .* W, 1.0 .* Matrix(I, p, p))) - p
    # value = tr(value) - p
    W2 = W .* W
    value = tr(W2 + W2 ^ 2 + W2 ^ 3 + W2 ^ 4 + W2 ^ 5) # this should be 0
    return value
end

function dnotears(W::Matrix{T}) where T
    println("dnotears is being called")
    return transpose(exp(W .* W)) .* W .* 2
end

function notears(d::ForwardDiff.Dual{T, V, P}) where {T, V, P}
    println("Forwardiff notears is being called")
    x = ForwardDiff.value(d)
    y = notears(x)
    dy = dnotears(x)
    return ForwardDiff.Dual{T}(y, dy * ForwardDiff.partials(d))
end

