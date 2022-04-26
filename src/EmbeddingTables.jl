module EmbeddingTables

#####
##### Exports
#####

# types
export AbstractEmbeddingTable, SimpleEmbedding, SplitEmbedding
export SparseEmbeddingUpdate, Static, Dynamic
export IndexingContext, NoContext, Forward, Update

export Indexer, DenseIndexer, SparseIndexer

# functions
export lookup, maplookup, featuresize, example, columnpointer

# strategies
export DefaultStrategy, SimpleParallelStrategy, PreallocationStrategy, Slicer

#####
##### Imports
#####

# deps
import ArrayInterface: ArrayInterface
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
import SIMD
import StaticArrays: StaticArrays, SVector, MVector
import Static: One, static, dynamic, StaticInt
import StrideArraysCore

# Execution strategies describe how to perform `maplookup` across an ensemble of embedding
# tables.
# The `DefaulfExecutionStrategy` defaults to serializing `lookup` across each table.
#
# This provides an entry point for developing specialized strategies.
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
Base.IndexStyle(::AbstractEmbeddingTable) = Base.IndexCartesian()

featuresize(A::AbstractMatrix) = size(A, 1)
featuresize(::AbstractEmbeddingTable{Static{N}}) where {N} = static(N)

abstract type IndexingContext end
struct NoContext <: IndexingContext end
struct Forward <: IndexingContext end
struct Update <: IndexingContext end

#####
##### columnpointer
#####

Base.@propagate_inbounds function columnpointer(A::AbstractMatrix{T}, i::Integer) where {T}
    return pointer(A) + strides(A)[2] * sizeof(T) * (i - 1)
end

# Implicitly remove "IndexingContext"
@inline function columnpointer(A::AbstractMatrix, i::Integer, ::IndexingContext)
    return columnpointer(A, i)
end

columnpointer(A::AbstractEmbeddingTable, ::Integer) =
    throw(ArgumentError("Please explicitly define `columnpointer` for $(typeof(A))"))

#####
##### ColumnView
#####

oneto(x::Integer) = Base.OneTo(x)
oneto(x::StaticInt) = One():x

# Bottom level `columnview`
@inline columnview(A::AbstractEmbeddingTable, len, i::Integer, ctx::IndexingContext) =
    StrideArraysCore.PtrArray(columnpointer(A, i, ctx), (len,))

@inline columnview(A::AbstractMatrix, len, i::Integer) = Base.unsafe_view(A, oneto(len), i)

@inline columnview(A::AbstractMatrix, len, i::Integer, _::IndexingContext) =
    columnview(A, len, i)

@inline columnview(A::AbstractMatrix, i::Integer, ctx::IndexingContext = NoContext()) =
    columnview(A, featuresize(A), i, ctx)

#####
##### example
#####

example(x::Vector{<:AbstractEmbeddingTable}) = example(first(x))

#####
##### ArrayInterface compat
#####

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
##### AbstractArray Interface
#####

Base.@propagate_inbounds function Base.getindex(A::AbstractEmbeddingTable, I::Vararg{Int,2})
    @boundscheck checkbounds(A, I...)
    return unsafe_load(columnpointer(A, I[2], NoContext()), I[1])
end

Base.@propagate_inbounds function Base.setindex!(
    A::AbstractEmbeddingTable,
    v,
    I::Vararg{Int,2},
)
    @boundscheck checkbounds(A, I...)
    return unsafe_store!(columnpointer(A, I[2], NoContext()), v, I[1])
end

#####
##### Implementation
#####

# utils
include("utils.jl")
include("simd.jl")

# interface
function lookup end
include("sparseupdate.jl")
include("lookup.jl")

# Embedding Table struct implementations
include("simple.jl")
include("split.jl")

end
