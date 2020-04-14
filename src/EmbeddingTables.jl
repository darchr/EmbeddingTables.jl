module EmbeddingTables

export SimpleEmbedding, lookup, maplookup

# For defining sparse adjoints.
using Flux
using Zygote

abstract type AbstractEmbeddingTable{T,N} <: AbstractArray{T,N} end

# Some generic interface implementations for AbstractEmbeddingTables
Base.IndexStyle(::AbstractEmbeddingTable) = Base.IndexLinear()

include("simple.jl")
include("update.jl")

end # module
