module EmbeddingTables

# types
export AbstractEmbeddingTable, SimpleEmbedding, SplitEmbedding
export SparseEmbeddingUpdate, UpdatePartitioner, Static, Dynamic

# functions
export lookup, maplookup

# strategies
export DefaultStrategy, SimpleParallelStrategy, PreallocationStrategy, Slicer

# deps
import ArrayInterface: ArrayInterface, static, dynamic, StaticInt, One
import ChainRulesCore: ChainRulesCore, NoTangent
import Dictionaries:
    Dictionaries,
    AbstractDictionary,
    Dictionary,
    gettoken!,
    gettoken,
    gettokenvalue,
    settokenvalue!
import Flux
import LoopVectorization
import ManualMemory
import Polyester
import StaticArrays: StaticArrays, SVector
import UnPack: @unpack

# Execution strategies describe how to perform `maplookup` across an ensemble of embedding
# tables.
# The `DefaulfExecutionStrategy` merely defaults to serializing `lookup` across each
# embedding table.
#
# This provides an entry point for developing strategies specialized for PMM
abstract type AbstractExecutionStrategy end
const VecOrMat{T} = Union{<:AbstractVector{T},<:AbstractMatrix{T}}

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

# Supertype for Embedding Tables
abstract type AbstractEmbeddingTable{S<:AbstractLookupType,T} <: AbstractArray{T,2} end

# Some generic interface implementations for AbstractEmbeddingTables
Base.IndexStyle(::AbstractEmbeddingTable) = Base.IndexLinear()

featuresize(A::AbstractMatrix) = size(A, 1)
Base.@propagate_inbounds function columnpointer(A::AbstractMatrix{T}, i::Integer) where {T}
    return pointer(A) + strides(A)[2] * sizeof(T) * (i - 1)
end

@inline columnview(A::AbstractMatrix, i::Integer) = columnview(A, axes(A, static(1)), i)
@inline columnview(A::AbstractMatrix, slice, i::Integer) = Base.unsafe_view(A, slice, i)
columnview(A::AbstractEmbeddingTable, _, ::Integer) =
    throw(ArgumentError("Please explicitly define `columnview` for $(typeof(A))"))
example(x::Vector{<:AbstractEmbeddingTable}) = example(first(x))

#####
##### ArrayInterface compat
#####

# ArrayInterface.can_change_size(::Type{<:AbstractEmbeddingTable}) = false
# ArrayInterface.can_setindex(::Type{<:AbstractEmbeddingTable}) = true
# ArrayInterface.contiguous_axis(::Type{<:AbstractEmbeddingTable}) = static(1)
#
# # In general, specific intantiations of embedding tables might not define strides,
# # especially if they are split into multiple subtables.
# ArrayInterface.defines_strides(::Type{<:AbstractEmbeddingTable}) = false
# ArrayInterface.fast_scalar_indexing(::Type{<:AbstractEmbeddingTable}) = false
# ArrayInterface.has_parent(::Type{<:AbstractEmbeddingTable}) = static(false)
# ArrayInterface.is_column_major(::Type{<:AbstractEmbeddingTable}) = static(true)

# Known sizing
ArrayInterface.known_size(::Type{<:AbstractEmbeddingTable{Static{N}}}) where {N} =
    (static(N), nothing)

function ArrayInterface.axes_types(::Type{<:AbstractEmbeddingTable{Static{N}}}) where {N}
    return Tuple{ArrayInterface.OptionallyStaticUnitRange{One,StaticInt{N}},Base.OneTo{Int}}
end

function Base.axes(A::AbstractEmbeddingTable{Static{N}}) where {N}
    return (One():static(N), Base.OneTo(size(A, 2)))
end

function ArrayInterface.size(A::AbstractEmbeddingTable{Static{N}}) where {N}
    return (static(N), Base.size(A, 2))
end

#####
##### Implementation
#####

# Interface
function lookup end
include("misc.jl")
include("sparseupdate.jl")
include("lookup.jl")

# Embedding Table struct implementations
include("simple.jl")
include("split.jl")

end
