mutable struct DAGProposal{P} <: AdvancedMH.Proposal{P}
    proposal::P
    accepted::Int64
    total   ::Int64
    X       ::Matrix{Float64}
    interventions::Matrix{Int8}
end

function DAGProposal(p::DAG, X, interventions) 
    DAGProposal(p, 0, 0, X, interventions)
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
    nv = size(t, 1) # genes 

    println("current transition is $t")
    x = deepcopy(t)
    perm = topological_sort(BitMatrix(x .!= 0))
    new_perm = deepcopy(perm)

    node_swap = rand(1:nv)
    local node_swap_partner

    if node_swap == 1
        node_swap_partner = 2
        @assert nv >= node_swap_partner && node_swap_partner >= 1
    elseif node_swap == nv
        node_swap_partner = nv - 1
        @assert nv >= node_swap_partner && node_swap_partner >= 1
    else
        node_swap_partner = node_swap + rand((-1, 1))
    end

    new_perm[node_swap] = perm[node_swap_partner]
    new_perm[node_swap_partner] = perm[node_swap]

    println("perm = $perm")
    println("new_perm = $new_perm")
    @assert new_perm != perm

    proposed_g = IMAP(new_perm, .01, proposal.X, proposal.interventions)
    println("proposed transition is $proposed_g")

    return proposed_g
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
    println("step here 1")
    candidate = AdvancedMH.propose(rng, sampler, model, transition_prev)
    println("step here 2")

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

function DynamicPPL.initialize_parameters!!(vi::AbstractVarInfo, init_params, spl::Sampler)
    @debug "Using passed-in initial variable values" init_params

    println("init_params = $init_params")
    # Flatten parameters.
    init_theta = mapreduce(vcat, init_params) do x
        vec([x;])
    end

    println("init_theta = $init_theta")
    # Get all values.
    linked = DynamicPPL.islinked(vi, spl)
    if linked
        # TODO: Make work with immutable `vi`.
        DynamicPPL.invlink!(vi, spl)
    end
    theta = vi[spl]

    println("length(theta) = $(length(theta))")
    println("length(init_theta) = $(length(init_theta))")

    length(theta) == length(init_theta) ||
        error("Provided initial value doesn't match the dimension of the model")


    # Update values that are provided.
    for i in 1:length(init_theta)
        x = init_theta[i]
        if x !== missing
            theta[i] = x
        end
    end

    # Update in `vi`.
    vi = DynamicPPL.setindex!!(vi, theta, spl)
    if linked
        # TODO: Make work with immutable `vi`.
        DynamicPPL.link!(vi, spl)
    end

    return vi
end
