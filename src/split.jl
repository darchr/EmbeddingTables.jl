# The split embedding table shards data into chunks.
# Each chunk is stored as a matrix.
struct SplitEmbedding{S,T,A<:AbstractMatrix{T}} <: AbstractEmbeddingTable{S,T}
    data::Vector{A}
    # All sub matrices (except for the last) should be the same size.
    # For the last, we provide some wiggle room to allow for non-perfectly sized
    # embedding tables.
    matrixsize::Tuple{Int,Int}

    # Inner constructor to ensure uniform sub-matrix sizing
    function SplitEmbedding(A::AbstractMatrix, cols_per_shard = 1)
        # Determine the number of shards we will have.
        nshards = ceil(Int, size(A, 2) / cols_per_shard)

        data = map(1:nshards) do i
            start = cols_per_shard * (i - 1) + 1
            stop = min(i * cols_per_shard, size(A, 2))

            B = similar(A, eltype(A), size(A, 1), stop - start + 1)
            @views B .= A[:, start:stop]
            return B
        end

        matrixsize = (size(A, 1), cols_per_shard)
        return new{Static{size(A, 1)},eltype(A),typeof(A)}(data, matrixsize)
    end

    # Undefined initializer
    function SplitEmbedding{S,T}(
        ::UndefInitializer,
        featuresize::Integer,
        ncols::Integer,
        cols_per_shard = 1;
        array::A = Vector{T}(),
        kw...,
    ) where {S,T,A}
        __compare(S, featuresize)
        nshards = ceil(Int, ncols / cols_per_shard)
        data = map(Base.OneTo(nshards)) do i
            start = cols_per_shard * (i - 1) + 1
            stop = min(i * cols_per_shard, ncols)
            return similar(array, T, (featuresize, stop - start + 1); kw...)
        end
        matrixsize = (featuresize, cols_per_shard)
        return new{S,T,eltype(data)}(data, matrixsize)
    end
end

__compare(::Type{Static{N}}, n) where {N} = @assert N == n
__compare(::Type{Dynamic}, n) = nothing

@inline _shardsize(A::SplitEmbedding) = A.matrixsize[2]

#####
##### Array Interface
#####

# Helper Functions
@inline _divrem_index(i, x) = _divrem_index(Int(i), Int(x))
@inline function _divrem_index(i::Int, x::Int)
    i -= 1
    a = Base.sdiv_int(i, x)
    b = Base.srem_int(i, x)
    return (a + 1), (b + 1)
end

@inline chunkindex(A::SplitEmbedding, i::Int) = _divrem_index(i, prod(A.matrixsize))

# Interface
function Base.size(A::SplitEmbedding)
    nrows = A.matrixsize[1]
    ncols = A.matrixsize[2] * (length(A.data) - 1) + size(last(A.data), 2)
    return (nrows, ncols)
end

# function Base.getindex(A::SplitEmbedding, i::Int)
#     @boundscheck checkbounds(A, i)
#     # Find which chunk the data is in, then lookup that chunk
#     chunk, index = chunkindex(A, i)
#     return @inbounds(A.data[chunk][index])
# end

# function Base.setindex!(A::SplitEmbedding, v, i::Int)
#     @boundscheck checkbounds(A, i)
#     # Find which chunk the data is in, then lookup that chunk
#     chunk, index = chunkindex(A, i)
#     return @inbounds(A.data[chunk][index] = v)
# end

#####
##### EmbeddingTables Interface
#####

example(A::SplitEmbedding) = first(A.data)
Base.@propagate_inbounds function columnpointer(A::SplitEmbedding, i::Integer)
    chunk, col = _divrem_index(i, _shardsize(A))
    @boundscheck checkbounds(A.data, chunk)
    data = @inbounds A.data[chunk]
    return columnpointer(data, col)
end

