# Ensure that the embedding styles are at least the same.
function update!(
    dst::AbstractEmbeddingTable{S,T},
    dst_bar::SparseEmbeddingUpdate{S},
    numcols::Integer,
    _alpha,
) where {S,T}
    alpha = convert(T, _alpha)
    src, indices = dst_bar.delta, dst_bar.indices
    for src_col in Base.OneTo(numcols)
        δ = columnview(src, axes(dst, static(1)), src_col)
        for dst_col in _maybe_columnview(indices, src_col)
            v = columnview(dst, dst_col)
            @inbounds for i in axes(dst, static(1))
                @_ivdep_meta
                @_interleave_meta(8)
                v[i] -= alpha * δ[i]
            end
        end
    end
    return nothing
end

#####
##### Flux Optimizers
#####

function Flux.Optimise.update!(
    opt,
    x,
    xbar::SparseEmbeddingUpdate,
    translations = Dict{Int,Int}();
    skip_crunch = false,
)
    if skip_crunch
        update!(x, xbar, length(xbar.indices), opt.eta)
    else
        update!(x, crunch(xbar, translations)..., opt.eta)
    end
    return nothing
end


