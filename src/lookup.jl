#####
##### Reference Implementations
#####

### Non-reducing
lookup(A::AbstractMatrix, I::AbstractVector{<:Integer}) = A[:, I]

### Reducing
function lookup(A::AbstractMatrix, II::AbstractMatrix{<:Integer})
    _A = [lookup(A, II[:, i]) for i = 1:size(II, 2)]
    _b = [[sum(_a[i, :]) for i = 1:size(_a, 1)] for _a in _A]
    return hcat(_b...)
end

#####
##### lookup
#####

_trailing_size(x::AbstractArray) = size(x)[end]

# Need these definitions to avoid method ambiguity
@inline lookup(A::AbstractEmbeddingTable, I::AbstractVector{<:Integer}) = _lookup(A, I)
@inline lookup(A::AbstractEmbeddingTable, I::AbstractMatrix{<:Integer}) = _lookup(A, I)

function _lookup(A::AbstractEmbeddingTable{S,T}, I::VecOrMat{<:Integer}) where {S,T}
    nrows = featuresize(A)
    O = similar(example(A), T, nrows, _trailing_size(I))
    # inner `lookup!` dispatches to either an optimized static or dynamic fallback
    # implementations.
    lookup!(O, A, I)
    return O
end

### Non-reducing

# Optimized static branch.
# Implementation is in "simd.jl"
function lookup!(
    dst, src::AbstractEmbeddingTable{Static{N},T}, indices::AbstractVector{<:Integer}
) where {T,N}
    cache_aligned_error(dst)
    #svec = SVector{N,T}
    for (dst_col, src_col) in enumerate(indices)
        @inbounds src_ptr = columnpointer(src, src_col)
        @inbounds dst_ptr = columnpointer(dst, dst_col)

        for i in 1:N
            @_ivdep_meta
            @_interleave_meta(8)
            unsafe_store!(dst_ptr, unsafe_load(src_ptr, i), i)
        end
    end
end

# fallback dynamic implementation
function lookup!(
    dst, src::AbstractEmbeddingTable{Dynamic,T}, indices::AbstractVector{<:Integer}
) where {T}
    nrows = featuresize(src)
    for (dst_col, src_col) in enumerate(indices)
        @inbounds src_ptr = columnpointer(src, src_col)
        @inbounds dst_ptr = columnpointer(dst, dst_col)

        for i in Base.OneTo(nrows)
            @_ivdep_meta
            @_interleave_meta(8)
            unsafe_store!(dst_ptr, unsafe_load(src_ptr, i), i)
        end
    end
end

### Reducing (sum)

# Optimized static branch.
# Implementation is in "simd.jl"
function lookup!(
    dst, src::AbstractEmbeddingTable{Static{N},T}, indices::AbstractMatrix{<:Integer}
) where {T,N}
    svec = SVector{N,T}
    sz = size(indices, 1)
    for dst_col in Base.OneTo(size(indices, 2))
        @_ivdep_meta

        # Move the first element to the destination, then sum.
        src_col = @inbounds(indices[1, dst_col])
        accum = unsafe_load(Ptr{svec}(columnpointer(src, src_col)))
        for offset = 2:sz
            src_col = @inbounds(indices[offset, dst_col])
            accum += unsafe_load(Ptr{svec}(columnpointer(src, src_col)))
        end
        unsafe_store!(Ptr{svec}(columnpointer(dst, dst_col)), accum)
    end
end

# fallback dynamic implementation
function lookup!(
    O, A::AbstractEmbeddingTable{Dynamic,T}, I::AbstractMatrix{<:Integer}
) where {T}
    nrows = featuresize(A)
    sz1, sz2 = size(I)
    for j in Base.OneTo(sz2)
        vO = columnview(O, j)
        # First iteration
        @inbounds col = I[1, j]
        vO .= columnview(A, col)

        # Accumulate all other times.
        for i in 2:sz1
            @inbounds col = I[i,j]
            vO .+= columnview(A, col)
        end
    end
    return nothing
end

#####
##### ColumnWrap
#####

# The `ColumnWrap` types lets us treat a 2D matrix as an array of arrays.
struct ColumnWrap{A}
    array::A
end

_colons(::AbstractArray{T,N}) where {T,N} = ntuple(_ -> :, Val(N-1))

unwrap(x::ColumnWrap) = x.array
Base.eachindex(x::ColumnWrap) = Base.OneTo(length(x))
Base.getindex(x::ColumnWrap, i::Integer) = view(x.array, _colons(x.array)..., i)
Base.length(x::ColumnWrap) = size(x.array, ndims(x.array))
function Base.iterate(x::ColumnWrap, i = 1)
    return in(i, eachindex(x)) ? (@inbounds(x[i]), i + 1) : nothing
end

# Dispatch plumbing for-the-win.
colwrap(x::ColumnWrap) = x
colwrap(x::AbstractVector{<:VecOrMat}) = x
colwrap(x::AbstractArray) = ColumnWrap(x)

#####
##### maplookup
#####

function maplookup(x::Vector{A}, i...) where {A<:AbstractEmbeddingTable}
    return maplookup(DefaultStrategy(), x, i...)
end

function ChainRulesCore.rrule(
    ::typeof(maplookup),
    strategy::AbstractExecutionStrategy,
    A::Vector{<:AbstractEmbeddingTable{S}},
    I,
) where {S}
    function maplookup_pullback(Δs)
        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            map(SparseEmbeddingUpdate{S}, Δs, colwrap(I)),
            ChainRulesCore.NoTangent(),
        )
    end
    result = maplookup(strategy, A, I)
    return result, maplookup_pullback
end

### Default Strategy
# Just all "lookup" on each table.
struct DefaultStrategy <: AbstractExecutionStrategy end
function maplookup(
    strategy::DefaultStrategy, x::Vector{A}, I
) where {A<:AbstractEmbeddingTable}
    return map(lookup, x, colwrap(I))
end

### Simple Parallel strategy.
# Thread lookups using Julia's normal multi-threading.
struct SimpleParallelStrategy <: AbstractExecutionStrategy end
function maplookup(::SimpleParallelStrategy, x::Vector{<:AbstractEmbeddingTable}, _I)
    out = Vector{typeof(example(x[1]))}(undef, length(x))

    # Need to capture this as a `ManualMemory.reference` to keep ManualMemory
    # from exploding while trying to store all the cached array stuff.
    I = ManualMemory.Reference(colwrap(_I))

    # Note - this is a hack to get around Polyester doing something weird!
    Polyester.@batch per=core for i in eachindex(x)
        out[i] = lookup(x[i], ManualMemory.dereference(I)[i])
    end
    return out
end

### Preallocate destinations
# The idea of the preallocation strategy is to essentially merge the "Concat" step with
# the "EmbeddingLookup" step.
#
# This may involve preallocating some space in the first few rows of the destination
# array to make space for inserting the result of the bottom MLP.
#
# TODO: Can we make the threading a little more finegrained for better threading?
struct PreallocationStrategy <: AbstractExecutionStrategy
    # Allow for extra rows to be placed at the beginning of the destination to allow
    # the results of dense computation to be inserted inplace.
    prependrows::Int
end
PreallocationStrategy() = PreallocationStrategy(0)

_batchsize(x::AbstractVector{<:AbstractVector}) = length(first(x))
_batchsize(x::AbstractMatrix{<:Integer}) = size(x, 1)

_batchsize(x::AbstractVector{<:AbstractMatrix}) = size(first(x), 2)
_batchsize(x::ColumnWrap) = _batchsize(unwrap(x))

function maplookup(
    strategy::PreallocationStrategy, x::Vector{<:AbstractEmbeddingTable{T}}, _I
) where {T}
    # Preallocate destination.
    I = ManualMemory.Reference(colwrap(_I))
    rows = featuresize.(x)
    offset = strategy.prependrows
    batchsize = _batchsize(_I)
    data = similar(example(x[1]), strategy.prependrows + sum(rows), batchsize)

    # For deciding where to index
    rows_sum = cumsum(rows)
    pushfirst!(rows_sum, 0)
    views = map(eachindex(x)) do i
        start = 1 + offset + rows_sum[i]
        stop = offset + rows_sum[i + 1]
        return view(data, start:stop, Base.OneTo(batchsize))
    end
    Polyester.@batch per=core for i in eachindex(x)
        lookup!(views[i], x[i], ManualMemory.dereference(I)[i])
    end
    return data
end

function ChainRulesCore.rrule(
    ::typeof(maplookup),
    strategy::PreallocationStrategy,
    A::Vector{<:AbstractEmbeddingTable{S}},
    _I,
) where {S}
    I = colwrap(_I)
    data = maplookup(strategy, A, I)
    function maplookup_pullback(Δ)
        f = Slicer(strategy.prependrows + 1, 1, Δ)
        δs = map((y, x) -> SparseEmbeddingUpdate{S}(f(featuresize(y)), x), A, I)
        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            δs,
            ChainRulesCore.NoTangent(),
        )
    end
    return data, maplookup_pullback
end
