module EmbeddingTables

# types
export AbstractEmbeddingTable, SimpleEmbedding, SplitEmbedding
export SparseEmbeddingUpdate, UpdatePartitioner, Static, Dynamic

# functions
export lookup, maplookup

# strategies
export DefaultStrategy, SimpleParallelStrategy, PreallocationStrategy, Slicer

# local deps
using CachedArrays

# deps
import ChainRulesCore
import DataStructures
import Flux
import ManualMemory
import Polyester
import SIMD
import StaticArrays: SVector
import UnPack: @unpack

# Execution strategies describe how to perform `maplookup` across an ensemble of embedding
# tables.
# The `DefaulfExecutionStrategy` merely defaults to serializing `lookup` across each
# embedding table.
#
# This provides an entry point for developing strategies specialized for PMM
abstract type AbstractExecutionStrategy end
const VecOrMat{T} = Union{<:AbstractVector{T}, <:AbstractMatrix{T}}

#####
##### Embedding Table API
#####

# Used to generate the lookup kernel.
#
# Static kernels have an unrolled kernel that generates significantly less code at the cost
# of requiring a fixed feature size.
abstract type AbstractLookupType end
struct Dynamic <: AbstractLookupType end
struct Static{N} <: AbstractLookupType end
Static(N) = Static{N}()

# For now, require nice alignment for static kernels.
const VECTOR_WIDTH_BYTES = 64
function require_cache_alignment(::Type{Static{N}}, ::Type{T}) where {N,T}
    rem = mod(sizeof(T) * N, VECTOR_WIDTH_BYTES)
    if !iszero(rem)
        msg = """
        Due to implementation limitations, the feature size for static lookup
        kernels must align to $VECTOR_WIDTH_BYTES bytes!

        For feature size $N, this is instead $(rem)!
        """
        throw(ArgumentError(msg))
    end
    return nothing
end

# Supertype for Embedding Tables
abstract type AbstractEmbeddingTable{S<:AbstractLookupType,T} <: AbstractArray{T,2} end
function require_cache_alignment(::AbstractEmbeddingTable{Static{N},T}) where {N,T}
    return require_cache_alignment(Static{N}, T)
end
require_cache_alignment(::AbstractEmbeddingTable{Dynamic}) = nothing

# Some generic interface implementations for AbstractEmbeddingTables
Base.IndexStyle(::AbstractEmbeddingTable) = Base.IndexLinear()

featuresize(A::AbstractMatrix) = size(A, 1)
featuresize(A::AbstractEmbeddingTable{Static{N}}) where {N} = N

Base.@propagate_inbounds function columnpointer(A::AbstractMatrix{T}, i::Integer) where {T}
    return pointer(A) + strides(A)[2] * sizeof(T) * (i - 1)
end
@inline columnview(A::AbstractMatrix, i) = Base.unsafe_view(A, Base.OneTo(featuresize(A)), i)

# Interface
function lookup end
include("simd.jl")
include("sparseupdate.jl")
include("slicer.jl")
include("lookup.jl")
include("update.jl")

# Embedding Table struct implementations
include("simple.jl")
include("split.jl")

# CachedArray Strategies
include("cachedarrays.jl")

end
