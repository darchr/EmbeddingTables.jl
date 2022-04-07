struct TiledSIMD{K,N,T}
    tiles::NTuple{K,SIMD.Vec{N,T}}
end

function simdtype(::Static{N}, ::Type{T}) where {N,T}
    vectorwidth = length(simdtype(T))
    K, r = divrem(N, vectorwidth)
    @assert iszero(r)
    return TiledSIMD{K,vectorwidth,T}
end
simdtype(::Type{Float32}) = SIMD.Vec{16,Float32}
simdtype(::Type{TiledSIMD{K,N,T}}) where {K,N,T} = SIMD.Vec{N,T}

Base.length(::Type{TiledSIMD{K,N,T}}) where {K,N,T} = K
Base.getindex(v::TiledSIMD{K,N,T}, i::Integer) where {K,N,T} = v.tiles[i]

@inline function Base.zero(::Type{TiledSIMD{K,N,T}}) where {K,N,T}
    tuple = ntuple(Val(K)) do _
        return zero(SIMD.Vec{N,T})
    end
    return TiledSIMD{K,N,T}(tuple)
end

@inline function load(::Type{TiledSIMD{K,N,T}}, ptr::Ptr{T}) where {K,N,T}
    tuple = ntuple(Val(K)) do i
        return SIMD.vload(SIMD.Vec{N,T}, ptr + sizeof(SIMD.Vec{N,T}) * (i - 1))
    end
    return TiledSIMD{K,N,T}(tuple)
end

@inline function store(
    v::TiledSIMD{K,N,T},
    ptr::Ptr{T},
    ::Val{Nontemporal} = Val(false),
) where {K,N,T,Nontemporal}
    return ntuple(Val(K)) do i
        return SIMD.vstore(
            @inbounds(v[i]),
            ptr + sizeof(SIMD.Vec{N,T}) * (i-1),
            nothing,
            Val(true),
            Val(Nontemporal),
        )
    end
end

@inline function Base.:+(a::TiledSIMD{K,N,T}, b::TiledSIMD{K,N,T}) where {K,N,T}
    tuple = ntuple(Val(K)) do i
        return a[i] + b[i]
    end
    return TiledSIMD{K,N,T}(tuple)
end

@inline function Base.muladd(x::T, y::TiledSIMD{K,N,T}, z::TiledSIMD{K,N,T}) where {T,K,N}
    tuple = ntuple(Val(K)) do i
        return muladd(x, y[i], z[i])
    end
    return TiledSIMD{K,N,T}(tuple)
end
