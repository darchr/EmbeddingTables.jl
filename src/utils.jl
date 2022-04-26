#####
##### Util Functions
#####

function zero!(x::AbstractArray{<:Any,T}) where {T}
    @inbounds for i in eachindex(x)
        x[i] = zero(T)
    end
end

#####
##### Compiler Hints
#####

# If we ever go back to using nontemporal stores
function sfence()
    str = raw"""
        tail call void asm sideeffect "sfence", "~{memory},~{dirflag},~{fpsr},~{flags}"()
        ret void
        """
    return Base.llvmcall(str, Nothing, Tuple{})
end

# Tells Julia that each iteration of the loop is independant.
# Useful for loops using raw loads and stores.
macro _ivdep_meta()
    return Expr(:loopinfo, Symbol("julia.ivdep"))
end

# Hint for LLVM - tell it how many iterations of a loop to unroll.
macro _interleave_meta(n)
    return Expr(:loopinfo, (Symbol("llvm.loop.interleave.count"), n))
end

macro _unroll_meta()
    return Expr(:loopinfo, Symbol("llvm.loop.unroll.enable"))
end

#####
##### Slicing
#####

# TODO: See if the #15276 style problems still exist in Julia 1.6.
#
# map` seems to be having Julia issue #15276 is problems when keeping track of where
# we are indexing to create views.
#
# As such, we have to build this `Slicer` struct below in order to give inference
# some help.
mutable struct Slicer{T,N,A<:AbstractArray{T,N}}
    current_index::Int
    concat_dim::Int
    captured_array::A
end

function (S::Slicer{T,N})(sz) where {T,N}
    current_index = S.current_index
    range = current_index:(current_index + sz - 1)
    inds = ntuple(i -> i == S.concat_dim ? range : 1:size(S.captured_array, i), Val(N))
    S.current_index += sz
    return view(S.captured_array, inds...)
end

#####
##### Utils for fast reverse indexing.
#####

struct ColumnIter{A<:AbstractMatrix}
    iter::A
end

columns(x::AbstractVector) = enumerate(x)
columns(x::AbstractMatrix) = ColumnIter(x)

function Base.iterate(x::ColumnIter, (col, row) = (1, 1))
    col > size(x.iter, 2) && return nothing
    item = @inbounds(x.iter[row, col])
    nextcol, nextrow = (row == featuresize(x.iter)) ? (col + 1, 1) : (col, row + 1)
    return (col, item), (nextcol, nextrow)
end
Base.length(x::ColumnIter) = length(x.iter)
Base.eltype(::Type{ColumnIter{A}}) where {T,A<:AbstractMatrix{T}} = Tuple{Int,T}

_maybe_columnview(x::AbstractVector, i, ::IndexingContext) = (x[i],)
_maybe_columnview(x::AbstractMatrix, i, ctx::IndexingContext) = columnview(x, i, ctx)

#####
##### Histograms
#####

struct OrderCount{T<:Integer}
    order::T
    count::T
end
Base.zero(::Type{OrderCount{T}}) where {T} = OrderCount(zero(T), zero(T))
# Note: indexed_iterate wants a "state" as the third argument to work properly, so
# add an unused third-argument to make this work.
Base.indexed_iterate(x::OrderCount, i::Int, _ = 1) = (getfield(x, i), i + 1)

struct ColOffset{T<:Integer}
    col::T
    offset::T
end
Base.zero(::Type{ColOffset{T}}) where {T} = ColOffset(zero(T), zero(T))
Base.indexed_iterate(x::ColOffset, i::Int, _ = 1) = (getfield(x, i), i + 1)

resize_histogram!(_, _) = nothing
resize_histogram!(array::AbstractVector, maxsize) = resize!(array, maxsize)

# Hack Warning - reaching into the internals of `Dictionaries.jl`
function shallow_empty!(dict::Dictionary)
    # Clear our indices
    indices = keys(dict)
    empty!(getfield(indices, :hashes))
    empty!(getfield(indices, :values))
    zero!(getfield(indices, :slots))
    setfield!(indices, :holes, 0)

    # Empty our values array
    empty!(Dictionaries._values(dict))
    return dict
end

function shallow_empty!(array::AbstractArray{T}) where {T}
    @inbounds for i in eachindex(array)
        array[i] = zero(T)
    end
end

function histogram!(histogram, A::AbstractArray)
    shallow_empty!(histogram)
    return unsafe_histogram!(histogram, A)
end

@inline function unsafe_histogram!(
    d::AbstractDictionary{K,V},
    A::AbstractArray,
) where {K<:Integer,V<:OrderCount}
    order = 0
    for a in A
        hasindex, token = gettoken!(d, a)
        if hasindex
            _order, _count = @inbounds gettokenvalue(d, token)
            @inbounds settokenvalue!(d, token, V(_order, _count + 1))
        else
            order += 1
            @inbounds settokenvalue!(d, token, V(order, 1))
        end
    end
    return order
end

@inline function unsafe_histogram!(
    histogram::AbstractArray{V},
    A::AbstractArray,
) where {V<:OrderCount}
    order = 0
    for a in A
        thisorder, thiscount = @inbounds(histogram[a])
        seenbefore = !iszero(thisorder)

        thisorder = ifelse(seenbefore, thisorder, order + 1)
        order = ifelse(seenbefore, order, order + 1)
        @inbounds histogram[a] = V(thisorder, thiscount + one(thiscount))
    end
    return order
end

### Prefix Sum
function prefixsum!(
    cumulative::AbstractVector{T},
    histogram::Dictionary,
    nnz::Integer = length(histogram),
) where {T<:ColOffset}
    resize!(cumulative, nnz + 1)
    next_offset = 1

    @inbounds for (k, (i, v)) in pairs(histogram)
        cumulative[i] = T(k, next_offset)
        next_offset += v
    end

    # Tailing terminator entry
    cumulative[end] = T(0, next_offset)
    return cumulative
end

function prefixsum!(
    cumulative::AbstractVector{T},
    histogram::AbstractVector,
    nnz::Integer,
) where {T<:ColOffset}
    resize!(cumulative, nnz + 1)

    # In this case, we need to perf
    @inbounds for (k, (i, v)) in pairs(histogram)
        # Skip entries in the histogram that haven't been seen before.
        iszero(i) && continue
        cumulative[i] = T(k, v)
    end

    next_offset = 1
    @inbounds for i in Base.OneTo(length(cumulative) - 1)
        col, count = cumulative[i]
        cumulative[i] = T(col, next_offset)
        next_offset += count
    end
    cumulative[end] = T(0, next_offset)
    return cumulative
end

### Remap
function remap!(
    map,
    cumulative,
    histogram::Dictionary{K,V},
    A::AbstractArray,
) where {K<:Integer,V<:OrderCount}
    resize!(map, length(A))
    @inbounds for (dst_col, src_col) in columns(A)
        _, token = gettoken(histogram, src_col)
        order, count = gettokenvalue(histogram, token)

        next_offset = cumulative[order + 1].offset
        map[next_offset - count] = dst_col
        settokenvalue!(histogram, token, V(order, count - one(count)))
    end
end

function remap!(
    map,
    cumulative,
    histogram::AbstractVector{V},
    A::AbstractArray,
) where {V<:OrderCount}
    resize!(map, length(A))
    @inbounds for (dst_col, src_col) in columns(A)
        order, count = histogram[src_col]
        next_offset = cumulative[order + 1].offset
        map[next_offset - count] = dst_col
        histogram[src_col] = V(order, count - one(count))
    end
end

#####
##### Indexers
#####

# Use an `AbstractIndexer` so we can potentially chunk up an actual `Indexer` for
# better parallel load balancing.
abstract type AbstractIndexer end
gettranslations(I::AbstractIndexer) = I.cumulative, I.map
gethistogram(I::AbstractIndexer) = I.histogram
Base.empty!(I::AbstractIndexer) = (shallow_empty!(gethistogram(I)); I)

const HistogramTypes =
    Union{Dictionary{<:Integer,<:OrderCount},AbstractVector{<:OrderCount}}

struct Indexer{T<:HistogramTypes} <: AbstractIndexer
    histogram::T
    # Mapping of source column to its start in the accumulation vector
    cumulative::Vector{ColOffset{Int}}
    map::Vector{Int}
end

const SparseIndexer = Indexer{Dictionary{Int,OrderCount{Int}}}
const DenseIndexer = Indexer{Vector{OrderCount{Int}}}

Indexer() = SparseIndexer()
function Indexer{T}() where {T}
    histogram = T()
    cumulative = Vector{ColOffset{Int}}()
    map = Vector{Int}()
    return Indexer(histogram, cumulative, map)
end

function index!(I::Indexer, A::AbstractArray, maxindex)
    (; histogram, cumulative, map) = I
    resize_histogram!(histogram, maxindex)

    nnz = histogram!(histogram, A)
    prefixsum!(cumulative, histogram, nnz)
    remap!(map, cumulative, histogram, A)
    return I
end

#####
##### View to chunk a full indexer into smaller pieces
#####

struct IndexerView <: AbstractIndexer
    I::Indexer
    range::UnitRange{Int}
end

function IndexerView(I::Indexer, num_splits, this_split)
    len = length(I.cumulative)
    split_size = cdiv(len, num_splits)

    start = (this_split - 1) * split_size + 1
    stop = min(this_split * split_size + 1, len)
    split_range = start:stop
    return IndexerView(I, split_range)
end

@inline function gettranslations(I::IndexerView)
    cumulative, map = gettranslations(I.I)
    return Base.unsafe_view(cumulative, I.range), map
end

