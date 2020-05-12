# Very simple, basic implementation of an embedding lookup.
struct SimpleEmbedding{T,A <: AbstractMatrix{T}} <: AbstractEmbeddingTable{T,2}
#struct SimpleEmbedding{T,A <: AbstractMatrix{T}} <: AbstractEmbeddingTable{T,2}
    data::A
end

# TODO: Do testing with `views` in Julia 1.5, since a lot of work has gone into making them
# faster and allocate less.
unsafe_column_pointer(A::SimpleEmbedding, i::Integer) = unsafe_column_pointer(A.data, i)

# Implement Array Interface
Base.size(A::SimpleEmbedding) = size(A.data)
Base.getindex(A::SimpleEmbedding, i::Int) = A.data[i]
Base.setindex!(A::SimpleEmbedding, v, i::Int) = (A.data[i] = v)

#lookup(A::SimpleEmbedding, I::Vector{<:Integer}) = A[:, I]
lookup(A::SimpleEmbedding, I) = _lookup(A.data, I)
_lookup(A::AbstractEmbeddingTable, I) = error("Called `_lookup` on an EmbeddingTable")
function _lookup(A::AbstractMatrix, I)
    nrows = size(A, 1)
    O = similar(A, eltype(A), nrows, length(I))
    @inbounds for (col, i) in enumerate(I)
        ptrA = unsafe_column_pointer(A, i)
        ptrO = unsafe_column_pointer(O, col)
        unsafe_copyto!(ptrO, ptrA, nrows)
    end
    return O
end
