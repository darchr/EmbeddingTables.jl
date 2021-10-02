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

struct ColumnIter{A <: AbstractMatrix}
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
Base.eltype(::Type{ColumnIter{A}}) where {T, A <: AbstractMatrix{T}} = Tuple{Int,T}

_maybe_columnview(x::AbstractVector, i) = (x[i],)
_maybe_columnview(x::AbstractMatrix, i) = columnview(x, i)

function histogram!(d::AbstractDictionary, A::AbstractArray)
    shallow_empty!(d)
    return unsafe_histogram!(d, A)
end

function unsafe_histogram!(d::AbstractDictionary, A::AbstractArray)
    Base.@_inline_meta
    order = 1
    for a in A
        hasindex, token = gettoken!(d, a)
        if hasindex
            _order, _count = @inbounds gettokenvalue(d, token)
            @inbounds settokenvalue!(d, token, (; order = _order, count = _count + 1))
        else
            @inbounds settokenvalue!(d, token, (; order = order, count = 1))
            order += 1
        end
    end
    return d
end

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

# Use an `AbstractIndexer` so we can potentially chunk up an actual `Indexer` for
# better parallel load balancing.
abstract type AbstractIndexer end
struct Indexer <: AbstractIndexer
    histogram::Dictionary{Int,NamedTuple{(:order, :count),Tuple{Int,Int}}}
    # Mapping of source column to its start in the accumulation vector
    cumulative::Vector{NamedTuple{(:col, :offset),Tuple{Int,Int}}}
    map::Vector{Int}
end

gettranslations(I::Indexer) = I.cumulative, I.map

function Indexer()
    histogram = Dictionary{Int,NamedTuple{(:order, :count),Tuple{Int,Int}}}()
    cumulative = Vector{NamedTuple{(:col, :offset),Tuple{Int,Int}}}()
    map = Vector{Int}()
    return Indexer(histogram, cumulative, map)
end

function Base.empty!(I::Indexer)
    shallow_empty!(I.histogram)
    return I
end

function prefixsum!(I::Indexer)
    cumulative, histogram = I.cumulative, I.histogram
    resize!(cumulative, length(histogram) + 1)
    next_offset = 1

    # The commented out routine is slightly faster, but has more allocations
    # @inbounds for (i, (k, (_, v))) in enumerate(pairs(histogram))
    #     cumulative[i] = (; col = k, offset = next_offset)
    #     next_offset += v
    # end

    @inbounds for (i, k) in enumerate(keys(histogram))
        v = histogram[k].count
        cumulative[i] = (; col = k, offset = next_offset)
        next_offset += v
    end

    # Tailing terminator entry
    cumulative[end] = (; col = 0, offset = next_offset)
    return cumulative
end

function index!(I::Indexer, A::AbstractArray)
    empty!(I)

    # Step 1 - create a histogram of values in the array.
    histogram = I.histogram
    unsafe_histogram!(histogram, A)

    # Step 2 - convert the histogram into the `cumulative` array that records the
    # cumulative sum of offsets
    cumulative = prefixsum!(I)

    # Step 3 (final pass) - Record the actual locations of columns in the update.
    map = I.map
    resize!(map, length(A))
    @inbounds for (column, x) in columns(A)
        _, token = gettoken(histogram, x)
        order, count = gettokenvalue(histogram, token)

        # Work backwards from the next offset
        next_offset = cumulative[order + 1].offset
        map[next_offset - count] = column
        settokenvalue!(histogram, token, (order = order, count = count - 1))
    end
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

function gettranslations(I::IndexerView)
    Base.@_inline_meta
    cumulative, map = gettranslations(I.I)
    return Base.unsafe_view(cumulative, I.range), map
end

