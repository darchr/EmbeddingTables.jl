#####
##### Compiler Hints
#####

# If we ever go back to using nontemporal stores
function sfence()
    str = raw"""
        tail call void asm sideeffect "sfence", "~{memory},~{dirflag},~{fpsr},~{flags}"()
        ret void
        """
    return Base.llvmcall(str, Nothing, Tuple{})
end

# Tells Julia that each iteration of the loop is independant.
# Useful for loops using raw loads and stores.
macro _ivdep_meta()
    return Expr(:loopinfo, Symbol("julia.ivdep"))
end

# Hint for LLVM - tell it how many iterations of a loop to unroll.
macro _interleave_meta(n)
    return Expr(:loopinfo, (Symbol("llvm.loop.interleave.count"), n))
end

macro _unroll_meta()
    return Expr(:loopinfo, Symbol("llvm.loop.unroll.enable"))
end

#####
##### Slicing
#####

# TODO: See if the #15276 style problems still exist in Julia 1.6.
#
# map` seems to be having Julia issue #15276 is problems when keeping track of where
# we are indexing to create views.
#
# As such, we have to build this `Slicer` struct below in order to give inference
# some help.
mutable struct Slicer{T,N,A<:AbstractArray{T,N}}
    current_index::Int
    concat_dim::Int
    captured_array::A
end

function (S::Slicer{T,N})(sz) where {T,N}
    current_index = S.current_index
    range = current_index:(current_index + sz - 1)
    inds = ntuple(
        i -> i == S.concat_dim ? range : 1:size(S.captured_array, i), Val(N)
    )
    S.current_index += sz
    return view(S.captured_array, inds...)
end
