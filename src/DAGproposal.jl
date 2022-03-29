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
    # println("debug me 1")
    d = DAG(deepcopy(t), 0.0, 0.05)
    # println("current t = $t")
    p = rand(rng, d)
    # println("current p = $p")
    return p
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
