# Very simple, basic implementation of an embedding lookup.
struct SimpleEmbedding{T,A <: AbstractMatrix{T}} <: AbstractEmbeddingTable{T,2}
#struct SimpleEmbedding{T,A <: AbstractMatrix{T}} <: AbstractEmbeddingTable{T,2}
    data::A
end

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
        ptrA = pointer(A, nrows * (i - 1) + 1)
        ptrO = pointer(O, nrows * (col - 1) + 1)
        unsafe_copyto!(ptrO, ptrA, nrows)
    end
    return O
end


