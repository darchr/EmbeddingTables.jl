alignment(::Type{T}, n) where {T} = n * sizeof(T)
isaligned(ptr::Ptr, n) = iszero(mod(UInt(ptr), n))

lastargs(x) = x
lastargs(x::Vector{Any}) = isempty(x) ? x : lastargs(x[end])
function lastargs(x::Expr)
    if x.head == :block
        return lastargs(x.args)
    elseif in(x.head, (:if, :elseif)) && length(x.args) == 3
        # recurse
        return lastargs(x.args[end])
    end
    return x.args
end

function ifelse_generator(f, nclauses)
    isfirst = true
    expr = Expr(:block)
    for i in Base.OneTo(nclauses)
        head = isfirst ? :if : :elseif
        clause, body = f(i)
        thisexpr = Expr(head, clause, body)
        push!(lastargs(expr), thisexpr)
        isfirst = false
    end
    return expr
end

function extract(::Type{SIMD.Vec{N,T}}, v::SVector{<:Any,T}, i) where {N,T}
    return SIMD.Vec{N,T}(ntuple(j -> @inbounds(v[j + i - 1]), Val(N)))
end

function ntstore_impl(N, ::Type{T}) where {T}
    # Select vector width based on alignment and overall length of `N`.
    # We need to ensure that whatever vector we choose
    vector_map = Dict(
        8 => 8,
        4 => 16,
        2 => 32,
        1 => 64,
    )
    base_vector_width = vector_map[sizeof(T)]

    # Generate code to deal with the incorrect leading alignment.
    # Assume that all native types are aligned correctly (e.g., Float32's are all correctly
    # aligned).
    nclauses = ceil(Int, log2(base_vector_width))
    # Build up the "if-else" tree
    vector_width = base_vector_width
    isfirst = true
    alignment_expr = ifelse_generator(nclauses) do i
        vector_width = div(vector_width, 2)
        clause = :(isaligned(ptr, $(alignment(T, vector_width))))
        body = quote
            SIMD.vstorent(extract(SIMD.Vec{$vector_width,$T}, v, base), ptr)
            ptr += sizeof(T) * $vector_width
            base += $vector_width
        end
        return (clause, body)
    end

    alignment_header = quote
        base = 1
        while !isaligned(ptr, $(alignment(T, base_vector_width)))
            $alignment_expr
        end
    end

    # Now the we have the alignment header, we generate the body of the loop.
    main = quote
        while ($(N + 1) - base) >= $(base_vector_width)
            SIMD.vstorent(extract(SIMD.Vec{$base_vector_width,$T}, v, base), ptr)
            base += $(base_vector_width)
            ptr += sizeof(T) * $base_vector_width
        end
    end

    vector_width = base_vector_width
    tail = ifelse_generator(nclauses) do _
        vector_width = div(vector_width, 2)
        clause = :($(N + 1) - base >= $vector_width)
        body = quote
            SIMD.vstorent(extract(SIMD.Vec{$vector_width,$T}, v, base), ptr)
            ptr += sizeof(T) * $vector_width
            base += $vector_width
        end
        return (clause, body)
    end

    return quote
        $alignment_header
        $main
        $tail
    end
end

@generated function ntstore!(ptr::Ptr{T}, v::SVector{N,T}) where {N,T}
    return ntstore_impl(N, T)
end

