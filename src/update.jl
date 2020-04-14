# A sparse updater for embedding tables.
struct SparseEmbeddingUpdate{I <: AbstractVector{<:Integer}, A <: AbstractArray}
    delta::A
    indices::I
end

# Apply the update
function Flux.Optimise.update!(x::AbstractArray, xbar::SparseEmbeddingUpdate)
    for (col, update) in zip(xbar.indices, eachcol(xbar.delta))
        @views x[:, col] .-= update
    end
end

# For now, just hijack a higher level of the Flux update chain.
function Flux.Optimise.update!(opt, x, xbar::SparseEmbeddingUpdate)
    return Flux.update!(x, Flux.Optimise.apply!(opt, x, xbar))
end

#####
##### Ajoints
#####

Zygote.@adjoint function lookup(A::AbstractEmbeddingTable, I)
    return lookup(A, I), Δ -> (SparseEmbeddingUpdate(Δ, I), nothing)
end

maplookup(x...) = lookup(x...)

# Overload the `_pullback` function so we can register A's gradient
#
# I really don't know why this is necessary, but it is.
# TODO: I'm pretty sure this is a Zygote bug.
function Zygote._pullback(cx::Zygote.AContext, ::typeof(maplookup), A::SimpleEmbedding, I)
    out = maplookup(A, I)
    return out, function pullback(Δ)
        ∇A = SparseEmbeddingUpdate(Δ, I)
        Zygote.accum_param(cx, A, ∇A)
        return (nothing, ∇A, nothing)
    end
end

#####
##### Optimizers
#####

function Flux.Optimise.apply!(opt::Flux.Descent, x, xbar::SparseEmbeddingUpdate)
    xbar.delta .*= opt.eta
    return xbar
end

