#####
##### Updates
#####

### Non-reduction case.
function Flux.Optimise.update!(
    x::AbstractEmbeddingTable,
    xbar::SparseEmbeddingUpdate{Dynamic,A,I},
    numcols::Integer,
    alpha,
) where {A,I<:AbstractVector}
    for (src_col, dst_col) in enumerate(view(xbar.indices, Base.OneTo(numcols)))
        # Update on a column-by-column basis
        update = columnview(xbar.delta, src_col)
        v = columnview(x, dst_col)
        v .-= alpha .* update
    end
    return nothing
end

function Flux.Optimise.update!(
    x::AbstractEmbeddingTable{Static{N},T},
    xbar::SparseEmbeddingUpdate{Static{N},T},
    numcols::Integer,
    _alpha
) where {N,T}
    alpha = convert(T, _alpha)
    svec = SVector{N,T}
    src = xbar.delta
    dst = x

    iter = view(xbar.indices, Base.OneTo(numcols))
    for (src_col, dst_col) in enumerate(iter)
        src_ptr = Ptr{svec}(columnpointer(src, src_col))
        dst_ptr = Ptr{svec}(columnpointer(dst, dst_col))

        v = unsafe_load(dst_ptr) - alpha * unsafe_load(src_ptr)
        unsafe_store!(dst_ptr, v)
    end
    return nothing
end

function Flux.Optimise.update!(
    x::AbstractEmbeddingTable{Static{N},T},
    xbar::SparseEmbeddingUpdate{Static{N},U},
    numcols::Integer,
    _alpha
) where {N,T,U}
    alpha = convert(T, _alpha)
    src = xbar.delta
    dst = x

    iter = view(xbar.indices, Base.OneTo(numcols))
    for (src_col, dst_col) in enumerate(iter)
        src_base = columnpointer(src, src_col)
        dst_base = columnpointer(dst, dst_col)
        for i in Base.OneTo(N)
            @_ivdep_meta
            @_interleave_meta(8)
            #before = unsafe_load(dst_base, i)
            v = unsafe_load(dst_base, i) - alpha * unsafe_load(src_base, i)
            unsafe_store!(dst_base, v, i)
        end
    end
    return nothing
end

#####
##### Optimizers
#####

function Flux.Optimise.update!(opt, x, xbar::SparseEmbeddingUpdate, args...)
    tup = Flux.Optimise.apply!(opt, x, xbar, args...)
    return Flux.update!(x, tup..., opt.eta)
end

# Prepare the SparseEmbeddingUpdate by first crunching it so we have less to write
# to the actual embedding table.
# During the crunch, we might as well multiply the gradiants by the learning rate
# since we're accessing
function Flux.Optimise.apply!(
    opt::Flux.Descent,
    x,
    xbar::SparseEmbeddingUpdate,
    translation = Dict{Int,Int}(),
)
    #eta = convert(eltype(xbar.delta), opt.eta)
    return crunch(xbar, translation)
end

