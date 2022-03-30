mutable struct DAGProposal{P} <: AdvancedMH.Proposal{P}
    proposal::P
    accepted::Int64
    total   ::Int64
end

function DAGProposal(p::DAG) 
    DAGProposal(p, 0, 0)
end

accepted!(p::DAGProposal) = p.accepted += 1
accepted!(p::Vector{<:DAGProposal}) = map(accepted!, p)
accepted!(p::NamedTuple{names}) where names = map(x->accepted!(getfield(p, x)), names)

# this is defined because the first draw has no transition yet (I think)
AdvancedMH.propose(rng::Random.AbstractRNG, p::DAGProposal, m::DensityModel) = 
    rand(rng, p.proposal)

# the actual proposal happens here
function AdvancedMH.propose(
    rng::Random.AbstractRNG,
    proposal::DAGProposal,
    model::DensityModel,
    t
)
    proposal.total += 1
    p = size(t, 1) # genes 

    max = 30
    count = 0

    orig = deepcopy(t)
    x = deepcopy(t)

    println("current transition is $t")
    
    non_zero_edges = findall(abs.(x) .> 0.0)
    reversed_edges = reverse.(non_zero_edges)
    zero_edges = findall(x .== 0.0)
    legal_new_edges = setdiff(zero_edges, reversed_edges)
    choices = ("delete", "reverse", "add")
    choice_dist = Categorical([1/3, 1/3, 1/3])
    local edge
    local reverse_edge
    sr = 0.0

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
        else # add new edge
            edge = sample(legal_new_edges)
            x[edge] = 1.0
        end
        
        sr = last(eigvals(x .* x))
        # println("sr = $sr")
        if (typeof(sr) <: Real) && (sr < 0.005)
            println("found a DAG; proposal = $proposal_choice, x = $x, count = $count, sr = $sr")
            return x
        else
            # println("not a DAG; proposal = $proposal_choice,  x = $x, count = $count, sr = $sr")
            x[edge] = orig[edge] # refer to original entry
            if @isdefined reverse_edge
                x[reverse_edge] = orig[reverse_edge] # refer to original entry
            end
        end
    end

    error("can't sample new DAG after $max attempts, sr = $sr")
end

function AdvancedMH.q(proposal::DAGProposal, t, t_cond) 
    # logpdf(proposal, t - t_cond)
    # logpdf(proposal, t - t_cond)
    return 0.0
end

# Define the other sampling steps.
# Return a 2-tuple consisting of the next sample and the the next state.
# In this case they are identical, and either a new proposal (if accepted)
# or the previous proposal (if not accepted).
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::AdvancedMH.MHSampler,
    transition_prev::AdvancedMH.AbstractTransition;
    kwargs...
)
    # Generate a new proposal.
    candidate = AdvancedMH.propose(rng, sampler, model, transition_prev)

    # Calculate the log acceptance probability and the log density of the candidate.
    logdensity_candidate = AdvancedMH.logdensity(model, candidate)
    loga = logdensity_candidate - AdvancedMH.logdensity(model, transition_prev) +
        AdvancedMH.logratio_proposal_density(sampler, transition_prev, candidate)

    # Decide whether to return the previous params or the new one.
    transition = if -Random.randexp(rng) < loga
	accepted!(sampler.proposal)
        # println("accepted!")
        AdvancedMH.transition(sampler, model, candidate, logdensity_candidate)
    else
        # println("rejected!")
        AdvancedMH.transition(sampler, model, candidate, logdensity_candidate)
        transition_prev
    end

    return transition, transition
end

accepted!(p::AdvancedMH.RandomWalkProposal) = nothing
accepted!(p::AdvancedMH.StaticProposal) = nothing
