module EmbeddingTables

export SimpleEmbedding, lookup, maplookup

# For defining sparse adjoints.
using Flux
using Zygote

abstract type AbstractEmbeddingTable{T,N} <: AbstractArray{T,N} end

# Some generic interface implementations for AbstractEmbeddingTables
Base.IndexStyle(::AbstractEmbeddingTable) = Base.IndexLinear()

unsafe_column_pointer(A::AbstractMatrix, i::Integer) = pointer(A, size(A,1) * (i-1) + 1)
columnview(A::AbstractMatrix, i) = view(A, :, i)

# Default reference implementation
lookup(A::AbstractMatrix, I) = A[:, I]

include("simple.jl")
include("split.jl")
include("update.jl")

end # module
