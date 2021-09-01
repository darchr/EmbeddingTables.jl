# Set to 64 for AVX-512
# Set to 32 for AVX2
# TODO: At some point, make this automatic?
function sfence()
    str = raw"""
        tail call void asm sideeffect "sfence", "~{memory},~{dirflag},~{fpsr},~{flags}"()
        ret void
        """
    return Base.llvmcall(str, Nothing, Tuple{})
end

function cache_aligned_error(A)
    if !iszero(mod(convert(UInt, pointer(A)), 64))
        error("Array must be aligned to a cache-line boundary (multiple of 64-bytes)!")
    end
end

#####
##### Loop Metadata
#####

# Tells Julia that each iteration of the loop is independant.
# Useful for loops using raw loads and stores.
macro _ivdep_meta()
    return Expr(:loopinfo, Symbol("julia.ivdep"))
end

# Hint for LLVM - tell it how many iterations of a loop to unroll.
macro _interleave_meta(n)
    return Expr(:loopinfo, (Symbol("llvm.loop.interleave.count"), n))
end

