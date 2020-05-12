# The split embedding table shards data into chunks.
# Each chunk is stored as a matrix.
struct SplitEmbedding{T,A <: AbstractMatrix{T}} <: AbstractEmbeddingTable{T,2}
    data::Vector{A}
    # All sub matrices (except for the last) should be the same size.
    # For the last, we provide some wiggle room to allow for non-perfectly sized
    # embedding tables.
    matrixsize::Tuple{Int,Int}

    # Inner constructor to ensure uniform sub-matrix sizing
    function SplitEmbedding(A::AbstractMatrix, cols_per_shard = 1)
        # Determine the number of shards we will have.
        nshards = ceil(Int, size(A,2) / cols_per_shard)

        data = map(1:nshards) do i
            start = cols_per_shard * (i - 1) + 1
            stop = min(i * cols_per_shard, size(A,2))
            return A[:, start:stop]
        end
        matrixsize = (size(A,1),cols_per_shard)

        return new{eltype(A),typeof(A)}(
            data,
            matrixsize,
        )
    end
end

# The size if the total size of the array.
function Base.size(A::SplitEmbedding)
    nrows = A.matrixsize[1]
    ncols = A.matrixsize[2] * (length(A.data) - 1) + size(last(A.data), 2)
    return (nrows, ncols)
end

function _divrem_index(i, x)
    a, b = divrem(i, x)
    # In the case where the remainder is zero, we actually need to step back one chunk.
    return iszero(b) ? (a, x) : (a+1, b)
end

function unsafe_column_pointer(A::SplitEmbedding, i::Integer)
    # Find the chunk and return the column from that chunk
    chunk, col = _divrem_index(i, A.matrixsize[2])
    return unsafe_column_pointer(A.data[chunk], col)
end

# Return a column view of some underlying chunk
function columnview(A::SplitEmbedding, i::Integer)
    chunk, col = _divrem_index(i, A.matrixsize[2])
    return columnview(A.data[chunk], col)
end

# Add 1 to the results to get into Index 1 land
chunkindex(A::SplitEmbedding, i::Int) = _divrem_index(i, prod(A.matrixsize))

function Base.getindex(A::SplitEmbedding, i::Int)
    @boundscheck checkbounds(A, i)
    # Find which chunk the data is in, then lookup that chunk
    chunk, index = chunkindex(A, i)
    return A.data[chunk][index]
end

function Base.setindex!(A::SplitEmbedding, v, i::Int)
    @boundscheck checkbounds(A, i)
    # Find which chunk the data is in, then lookup that chunk
    chunk, index = chunkindex(A, i)
    return A.data[chunk][index] = v
end

function lookup(A::SplitEmbedding, I)
    nrows = size(A, 1)

    like = first(A.data)
    O = similar(like, eltype(like), nrows, length(I))
    @inbounds for (col, i) in enumerate(I)
        ptrA = unsafe_column_pointer(A, i)
        ptrO = unsafe_column_pointer(O, col)
        unsafe_copyto!(ptrO, ptrA, nrows)
    end
    return O
end

