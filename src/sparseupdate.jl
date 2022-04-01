#####
##### Embedding Updates
#####

# A sparse updater for embedding tables.
struct SparseEmbeddingUpdate{S<:AbstractLookupType,A<:AbstractMatrix,I<:AbstractArray}
    delta::A
    indices::I
end

function SparseEmbeddingUpdate{S}(delta::A, indices::I) where {S,A,I}
    return SparseEmbeddingUpdate{S,A,I}(delta, indices)
end

# Convert the compressed representations
function uncompress(
    x::SparseEmbeddingUpdate,
    dstcols = maximum(x.indices);
    maxindices = length(x.indices),
)
    indices, delta = x.indices, x.delta
    dst = zeros(eltype(delta), size(delta, 1), dstcols)
    count = 0
    for (column, update) in enumerate(eachcol(delta))
        for c in _maybe_columnview(indices, column, Update())
            columnview(dst, c, Update()) .+= update
        end
        count += 1
        count == maxindices && break
    end
    return dst
end

# -- pullback
function ChainRulesCore.rrule(::typeof(lookup), A::AbstractEmbeddingTable{S}, I) where {S}
    function lookup_pullback(Δ)
        return (NoTangent(), SparseEmbeddingUpdate{S}(Δ, I), NoTangent())
    end
    return lookup(A, I), lookup_pullback
end

#####
##### Accumulate and Update
#####

function update!(
    table::AbstractEmbeddingTable{S,T},
    update::SparseEmbeddingUpdate{S},
    indexer::AbstractIndexer,
    alpha,
    ::Val{Nontemporal};
    kw...,
) where {S,T,Nontemporal}
    return _update_generic_impl!(table, update, indexer, alpha, Val(Nontemporal); kw...)
end

function _update_generic_impl!(
    table::AbstractEmbeddingTable{S,T},
    update::SparseEmbeddingUpdate{S},
    indexer::AbstractIndexer,
    alpha,
    ::Val{Nontemporal} = Val(true);
    scratchspace::AbstractVector = Vector{T}(undef, featuresize(table)),
) where {S,T,Nontemporal}
    Base.@_inline_meta
    @assert length(scratchspace) == featuresize(table)

    grads = update.delta
    cumulative, map = gettranslations(indexer)
    @inbounds for entry in Base.OneTo(length(cumulative) - 1)
        k, start = cumulative[entry]
        stop = cumulative[entry + 1].offset - 1
        zero!(scratchspace)

        i = start
        while i <= stop
            col = map[i]
            cv = columnview(grads, featuresize(table), col, Update())
            @inbounds for i in axes(table, static(1))
                @_ivdep_meta
                @_interleave_meta(8)
                scratchspace[i] += cv[i]
            end
            i += 1
        end

        # Update using nontemporal stores.
        tableview = columnview(table, k, Update())
        f(x, y) = x - alpha * y
        if Nontemporal
            LoopVectorization.vmapnt!(f, tableview, tableview, scratchspace)
        else
            LoopVectorization.vmap!(f, tableview, tableview, scratchspace)
        end
    end
end

function _update_specialized_impl!(
    table::AbstractEmbeddingTable{Static{N},T},
    update::SparseEmbeddingUpdate{Static{N}},
    indexer::AbstractIndexer,
    alpha,
    ::Val{Nontemporal} = Val(true);
    kw...,
) where {N,T,Nontemporal}
    Base.@_inline_meta

    grads = update.delta
    cumulative, map = gettranslations(indexer)
    @inbounds for entry in Base.OneTo(length(cumulative) - 1)
        k, start = cumulative[entry]
        stop = cumulative[entry + 1].offset - 1

        accum = zeros(SVector{N,T})
        i = start
        while i <= stop
            col = map[i]
            accum += unsafe_load(Ptr{SVector{N,T}}(columnpointer(grads, col, Update())))
            i += 1
        end

        # Done accumulating - time to write back out update.
        #
        # Need to manually specifiy that the pointer address for `accum` is stable because
        # LoopVectorization's automatic treatment doesn't preserve a pointer to `accum`
        # correctly.
        #
        # Julia is able to elide the creation of the `Ref` altogether.
        tableview = columnview(table, k, Update())
        ref = Ref(accum)
        GC.@preserve ref begin
            ptrarray =
                StrideArraysCore.PtrArray(Base.unsafe_convert(Ptr{T}, ref), (static(N),))

            if Nontemporal
                LoopVectorization.vmapnt!(
                    (x, y) -> x - alpha * y,
                    tableview,
                    tableview,
                    ptrarray,
                )
            else
                LoopVectorization.vmap!(
                    (x, y) -> x - alpha * y,
                    tableview,
                    tableview,
                    ptrarray,
                )
            end
        end
    end
end

@generated function update!(
    table::AbstractEmbeddingTable{Static{N},T},
    update::SparseEmbeddingUpdate{Static{N}},
    indexer::AbstractIndexer,
    alpha,
    ::Val{Nontemporal} = Val(true);
    kw...,
) where {N,T,Nontemporal}
    bytes = N * sizeof(T)
    # Slightly smaller heuristic for all-registers.
    if bytes <= div(MAX_ACCUMULATOR_SIZE, 2)
        # No need to forward any keywords to the specialized implementation.
        return :(_update_specialized_impl!(table, update, indexer, alpha, Val(Nontemporal)))
    else
        return :(_update_generic_impl!(
            table,
            update,
            indexer,
            alpha,
            Val(Nontemporal);
            kw...,
        ))
    end
end

#####
##### Flux Compat
#####

function update!(
    opt::Flux.Descent,
    table::AbstractEmbeddingTable,
    update::SparseEmbeddingUpdate,
    indexer = Indexer(),
    ::Val{Nontemporal} = Val(true);
    kw...,
) where {Nontemporal}
    index!(indexer, update.indices)
    update!(
        table,
        update,
        indexer,
        convert(eltype(table), opt.eta),
        Val(Nontemporal);
        kw...,
    )
    return nothing
end

function Flux.Optimise.update!(
    opt,
    x,
    xbar::SparseEmbeddingUpdate,
    indexer = Indexer(),
    ::Val{Nontemporal} = Val(true);
    kw...,
) where {Nontemporal}
    return update!(opt, x, xbar, indexer, Val(Nontemporal); kw...)
end

#####
##### Ensemble Update
#####

ensemble_update(nthreads::Integer) = [Indexer() for _ = 1:nthreads]
scratch(table::AbstractEmbeddingTable) =
    (_...) -> Vector{eltype(table)}(undef, featuresize(table))

function update!(
    opt::Flux.Descent,
    tables::AbstractVector{<:AbstractEmbeddingTable},
    grads::AbstractVector{<:SparseEmbeddingUpdate},
    indexers::Vector{Indexer},
    ::Val{Nontemporal};
    num_splits = 4,
    nthreads = Threads.nthreads(),
    scratchspaces = map(scratch(first(tables)), 1:Threads.nthreads()),
) where {Nontemporal}
    # First, index all of the tables.
    Polyester.@batch (per = thread) for i in eachindex(indexers, grads)
        index!(indexers[i], grads[i].indices)
    end

    # Now - do the work of updating
    len = num_splits * length(tables)
    count = Threads.Atomic{Int}(1)
    Polyester.@batch (per = thread) for tid in Base.OneTo(nthreads)
        while true
            k = Threads.atomic_add!(count, 1)
            k > len && break

            i, j = _divrem_index(k, num_splits)
            indexer_view = IndexerView(indexers[i], num_splits, j)
            scratchspace = scratchspaces[tid]
            update!(
                tables[i],
                grads[i],
                indexer_view,
                opt.eta,
                Val(Nontemporal);
                scratchspace = scratchspace,
            )
        end
    end
end

