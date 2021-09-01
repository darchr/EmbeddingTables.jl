CachedArrays.@wrapper SimpleEmbedding data
CachedArrays.@wrapper EmbeddingTables.SparseEmbeddingUpdate (unsafe_free,) delta

function CachedArrays.constructorof(::Type{<:SimpleEmbedding{Static{N}}}) where {N}
    return SimpleEmbedding{Static{N}}
end

@annotate function EmbeddingTables.lookup!(
    O, A::SimpleEmbedding{S,T,<:UnreadableCachedArray}, I::AbstractVector{<:Integer}
) where {S,T,N}
    return __recurse__(O, __readable__(A), I)
end

@annotate function EmbeddingTables.lookup!(
    O, A::SimpleEmbedding{S,T,<:UnreadableCachedArray}, I::AbstractMatrix{<:Integer}
) where {S,T,N}
    return __recurse__(O, __readable__(A), I)
end

@annotate function Flux.update!(
    x::EmbeddingTables.SimpleEmbedding{Static{N},T,<:UnwritableCachedArray},
    xbar::EmbeddingTables.SparseEmbeddingUpdate{Static{N},<:AbstractMatrix,<:AbstractVector},
    numcols::Integer,
) where {N,T}
    return __recurse__(__writable__(x), xbar, numcols)
end

#####
##### Constructors
#####

# function make_tables(
#     sizes::AbstractVector, featuresize::Int, manager::CachedArrays.CacheManager
# )
#     constructor = tocached(
#         Float32, manager, CachedArrays.ReadWrite(); priority = CachedArrays.ForceRemote
#     )
#     function init(args...)
#         data = constructor(args...)
#         _Model.multithread_init(_Model.ZeroInit(), data)
#         return data
#     end
#
#     f = x -> SplitEmbedding(x, 1024)
#     g = SplitEmbedding{Static{featuresize}}
#
#     return _Model.create_embeddings(g, featuresize, sizes, init)
# end

function make_updates(
    sizes::AbstractVector,
    featuresize::Int,
    manager::CachedArrays.CacheManager;
    indices_per_result = 1,
    batchsize = 16,
)
    constructor = tocached(Float32, manager, CachedArrays.ReadWrite())
    function init(args...)
        data = constructor(args...)
        Random.randn!(data)
        return data
    end

    arrays = [init(featuresize, batchsize) for _ in sizes]
    if indices_per_result == 1
        indices = [rand(1:sz, batchsize) for sz in sizes]
    else
        indices = [rand(1:sz, indices_per_result, batchsize) for sz in sizes]
    end
    return (SparseEmbeddingUpdate{Static{featuresize}}).(arrays, indices)
end

function histogram!(h::Dict{T,Int}, x::AbstractArray{T}) where {T}
    empty!(h)
    for i in x
        h[i] = get(h, i, 0) + 1
    end
    return h
end
