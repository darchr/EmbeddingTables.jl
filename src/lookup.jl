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

# Maximum amount of state we will keep in the CPU registers during accumulation.
# If necessary, we will make multiple passes for larger feature sizes.
const MAX_ACCUMULATOR_SIZE = 1024

_trailing_size(x::AbstractArray{<:Any,N}) where {N} = size(x, N)

# Need these definitions to avoid method ambiguity
@inline lookup(A::AbstractEmbeddingTable, I::AbstractVector{<:Integer}) = _lookup(A, I)
@inline lookup(A::AbstractEmbeddingTable, I::AbstractMatrix{<:Integer}) = _lookup(A, I)

function _lookup(A::AbstractEmbeddingTable, I::VecOrMat{<:Integer})
    nrows = featuresize(A)
    O = similar(example(A), eltype(A), nrows, _trailing_size(I))
    lookup!(O, A, I)
    return O
end

#####
##### Non-reducing
#####

# Generic fallback case
function lookup!(dst, src::AbstractEmbeddingTable, indices::AbstractVector{<:Integer})
    for (dst_col, src_col) in enumerate(indices)
        @inbounds src_view = columnview(src, src_col, Forward())
        @inbounds dst_view = columnview(dst, dst_col, Forward())

        @inbounds for i in ArrayInterface.axes(src, static(1))
            @_ivdep_meta
            @_interleave_meta(8)
            dst_view[i] = src_view[i]
        end
    end
end

# single-shot load + store up to MAX_ACCUMULATOR_SIZE
function _lookup_kernel!(
    ::Type{SVector{N,T}},
    dst,
    src::AbstractEmbeddingTable,
    indices::AbstractVector{<:Integer},
) where {N,T}
    Base.@_inline_meta
    @inbounds for dst_col in axes(dst, 2)
        @_ivdep_meta
        src_col = indices[dst_col]
        src_ptr = convert(Ptr{SVector{N,T}}, columnpointer(src, src_col, Forward()))
        dst_ptr = convert(Ptr{SVector{N,T}}, columnpointer(dst, dst_col, Forward()))
        unsafe_store!(dst_ptr, unsafe_load(src_ptr))
    end
    return nothing
end

@generated function lookup!(
    dst,
    src::AbstractEmbeddingTable{Static{N},T},
    indices::AbstractVector{<:Integer},
) where {N,T}
    Base.@_inline_meta
    # First, check if we can do this in a single shot.
    # If so, do that.
    # Otherwise, invoke the generic fallback.
    bytes = N * sizeof(T)
    if bytes <= MAX_ACCUMULATOR_SIZE
        return :(_lookup_kernel!(SVector{N,T}, dst, src, indices))
    else
        return quote
            Base.invoke(
                lookup!,
                Tuple{$dst,AbstractEmbeddingTable,$indices},
                dst,
                src,
                indices
           )
        end
    end
end

#####
##### Reducing (sum)
#####

function _reducing_lookup_kernel(
    ::Type{SVector{N,T}},
    table::AbstractEmbeddingTable,
    indices::AbstractVector,
) where {N,T}
    Base.@_inline_meta
    col = @inbounds indices[1]
    # Move the first element into registers
    ptr = convert(Ptr{SVector{N,T}}, columnpointer(table, col, Forward()))
    accumulator = unsafe_load(ptr)
    for i = 2:(lastindex(indices))
        @_ivdep_meta
        col = @inbounds indices[i]
        ptr = convert(Ptr{SVector{N,T}}, columnpointer(table, col, Forward()))
        accumulator += unsafe_load(ptr)
    end
    return accumulator
end

function _reducing_lookup_kernel!(
    ::Type{SVector{N,T}},
    dst,
    src::AbstractEmbeddingTable,
    indices::AbstractMatrix{<:Integer},
) where {N,T}
    Base.@_inline_meta
    for dst_col in axes(dst, 2)
        accumulator = _reducing_lookup_kernel(
            SVector{N,T},
            src,
            Base.unsafe_view(indices, axes(indices, static(1)), dst_col),
        )
        dst_ptr = Ptr{SVector{N,T}}(columnpointer(dst, dst_col, Forward()))
        unsafe_store!(dst_ptr, accumulator)
    end
    return nothing
end

@generated function lookup!(
    dst,
    src::AbstractEmbeddingTable{Static{N},T},
    indices::AbstractMatrix{<:Integer},
) where {N,T}
    Base.@_inline_meta
    # First , check if we can hold all partial state in registers.
    # If so, emit the simplest kernel possible.
    # Otherwise, just fall back to the generic implementation.
    # The static sizing information will carry over to the code generation.
    bytes = N * sizeof(T)
    if bytes <= MAX_ACCUMULATOR_SIZE
        return :(_reducing_lookup_kernel!(SVector{N,T}, dst, src, indices))
    else
        return quote
            Base.invoke(
                lookup!,
                Tuple{$dst,AbstractEmbeddingTable,$indices},
                dst,
                src,
                indices
           )
        end
    end
end

# fallback implementation
function lookup!(
    O,
    A::AbstractEmbeddingTable,
    I::AbstractMatrix{<:Integer},
)
    for j in axes(I, 2)
        vO = columnview(O, featuresize(A), j, Forward())
        # First iteration
        col = @inbounds I[1, j]
        vA = columnview(A, col, Forward())
        @inbounds for k in axes(A, 1)
            @_ivdep_meta
            @_interleave_meta(8)
            vO[k] = vA[k]
        end

        # Accumulate all other times.
        for i in 2:size(I, 1)
            col = @inbounds I[i, j]
            vA = columnview(A, col, Forward())
            @inbounds for k in axes(A, 1)
                @_ivdep_meta
                @_interleave_meta(8)
                vO[k] += vA[k]
            end
        end
    end
    return nothing
end

############################################################################################
# Multiple table lookups
############################################################################################

#####
##### ColumnWrap
#####

# The `ColumnWrap` types lets us treat a 2D matrix as an array of arrays.
struct ColumnWrap{A}
    array::A
end

@inline _colons(::AbstractArray{T,N}) where {T,N} = ntuple(_ -> :, Val(N - 1))
@inline _colons(x::ColumnWrap) = _colons(unwrap(x))

unwrap(x::ColumnWrap) = x.array
Base.eachindex(x::ColumnWrap) = Base.OneTo(length(x))
Base.getindex(x::ColumnWrap, i::Integer) = view(unwrap(x), _colons(x)..., i)
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

### Default Strategy
struct DefaultStrategy <: AbstractExecutionStrategy end
maplookup(x::Vector{<:AbstractEmbeddingTable}, i...) = maplookup(DefaultStrategy(), x, i...)
function maplookup(strategy::DefaultStrategy, x::Vector{<:AbstractEmbeddingTable}, I)
    return maplookup_impl(strategy, x, colwrap(I))
end

function maplookup_impl(_::DefaultStrategy, x::Vector{<:AbstractEmbeddingTable}, I)
    return map(lookup, x, colwrap(I))
end

# Generic pullback
# Inform `ChainRulesCore` that `SparseEmbeddingUpdates` are suitable differentials for
# `AbstractArray`s.
(p::ChainRulesCore.ProjectTo{AbstractArray, <:NamedTuple})(x::SparseEmbeddingUpdate) = x
function ChainRulesCore.rrule(
    ::typeof(maplookup_impl),
    strategy::AbstractExecutionStrategy,
    A::Vector{<:AbstractEmbeddingTable{S}},
    I,
) where {S}
    result = maplookup(strategy, A, I)
    function maplookup_pullback(Δs)
        return (NoTangent(), NoTangent(), map(SparseEmbeddingUpdate{S}, Δs, I), NoTangent())
    end
    return result, maplookup_pullback
end

### Simple Parallel strategy.
# Thread lookups using Polyester.
struct SimpleParallelStrategy <: AbstractExecutionStrategy end
function maplookup(strategy::SimpleParallelStrategy, x::Vector{<:AbstractEmbeddingTable}, I)
    return maplookup_impl(strategy, x, colwrap(I))
end

function maplookup_impl(_::SimpleParallelStrategy, x::Vector{<:AbstractEmbeddingTable}, I)
    out = Vector{typeof(example(x[1]))}(undef, length(x))

    # Need to capture this as a `ManualMemory.reference` to keep ManualMemory
    # from exploding while trying to store all the cached array stuff.
    ref = ManualMemory.Reference(I)
    Polyester.@batch (per = core) for i in eachindex(x)
        out[i] = lookup(x[i], ManualMemory.dereference(ref)[i])
    end
    return out
end

### Preallocate destinations
# The idea of the preallocation strategy is to essentially merge the "Concat" step with
# the "EmbeddingLookup" step.
#
# This may involve preallocating some space in the first few rows of the destination
# array to make space for inserting the result of the bottom MLP.
struct PreallocationStrategy{T} <: AbstractExecutionStrategy
    # Allow for extra rows to be placed at the beginning of the destination to allow
    # the results of dense computation to be inserted inplace.
    prependrows::Int
end

PreallocationStrategy() = PreallocationStrategy{Any}(0)
PreallocationStrategy(x::Integer) = PreallocationStrategy{Any}(x)

_select_eltype(::Type{Any}, ::Type{T}) where {T} = T
_select_eltype(::Type{U}, ::Type{T}) where {U,T} = U

_batchsize(x::AbstractVector{<:AbstractArray}) = _trailing_size(first(x))
_batchsize(::AbstractVector{<:Integer}) = 1
_batchsize(x::AbstractArray{<:Integer,N}) where {N} = size(x, N-1)
_batchsize(x::ColumnWrap) = _batchsize(unwrap(x))

cdiv(a::Integer, b::Integer) = cdiv(promote(a, b)...)
cdiv(a::T, b::T) where {T<:Integer} = one(T) + div(a - one(T), b)
viewlast(x::AbstractArray, inds) = view(x, _colons(x)..., inds)

function maplookup(strategy::PreallocationStrategy, x::Vector{<:AbstractEmbeddingTable}, I)
    return maplookup_impl(strategy, x, colwrap(I))
end

function maplookup_impl(
    strategy::PreallocationStrategy{U},
    x::Vector{<:AbstractEmbeddingTable{<:Any,T}},
    I,
) where {U,T}
    worksize_div = 8

    # Preallocate destination.
    ref = ManualMemory.Reference(I)
    rows = map(featuresize, x)
    offset = strategy.prependrows
    batchsize = _batchsize(I)
    worksize = cdiv(batchsize, worksize_div)

    ncols = strategy.prependrows + sum(rows)
    data = similar(example(x), _select_eltype(U, T), (ncols, batchsize))

    # Chunk up the destination array into pieces.
    # Each piece serves as the destination for an embedding table lookup.
    rows_sum = cumsum(rows)
    pushfirst!(rows_sum, 0)
    views = map(eachindex(x)) do i
        start = 1 + offset + rows_sum[i]
        stop = offset + rows_sum[i + 1]
        return view(data, start:stop, Base.OneTo(batchsize))
    end

    # Implement the poor man's dynamic load balancing.
    len = worksize_div * length(x)
    divisor = length(x)
    count = Threads.Atomic{Int}(1)
    Polyester.@batch (per = core) for _ in Base.OneTo(Threads.nthreads())
        while true
            k = Threads.atomic_add!(count, 1)
            k > len && break

            # Convert this in to a big and little index
            # Parallelize first across tables, then within a table.
            # This can reduce the number of threads working on a single table, which
            # can be helpful is synchronization is required.
            j, i = _divrem_index(k, divisor)

            # Compute the start and the stop indices along the batch dimension.
            start = (j - 1) * worksize + 1
            stop = min(j * worksize, batchsize)

            # Slice along the batch dimension
            v = view(views[i], :, start:stop)
            table = x[i]
            indices = viewlast(ManualMemory.dereference(ref)[i], start:stop)

            # Finally, perform the final lookup
            lookup!(v, table, indices)
        end
    end
    return data
end

# Specialized `rrule` that knows how to deal with the pre-pended columns.
function ChainRulesCore.rrule(
    ::typeof(maplookup_impl),
    strategy::PreallocationStrategy,
    A::Vector{<:AbstractEmbeddingTable{S}},
    I,
) where {S}
    data = maplookup(strategy, A, I)
    function maplookup_pullback(Δ)
        f = Slicer(strategy.prependrows + 1, 1, Δ)
        δs = map((y, x) -> SparseEmbeddingUpdate{S}(f(featuresize(y)), x), A, I)
        return (NoTangent(), NoTangent(), δs, NoTangent())
    end
    return data, maplookup_pullback
end

