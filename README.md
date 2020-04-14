# EmbeddingTables

Support for Embedding Tables with sparse updates compatible with the Flux and Zygote ecosystem.

Optimizer Support: `Flux.Descent`

## Technical Notes

Zygote support performing this sparse update is a little weird and requires digging a little into the internal of the Zygote auto-diff framework.
Normally, one would define something like
```julia
Zygote.@adjoint lookup(A::SimpleEmbeding,I) =
    lookup(A,I), Δ -> (SparseEmbeddingUpdate(Δ,I), nothing)
```
which works fine.

However, the problem comes when we have a collection of embedding tables and try to lookup indices from each table using `map`.
In this scenario, Zygote does not capture the implicitly used embedding tables in the vector and simply returns `nothing` for the gradient.

We get around this by going one level deeper:
```julia
function Zygote._pullback(cx::Zygote.AContext, ::typeof(lookup), A::SimpleEmbedding, I)
    out = lookup(A, I)
    return out, function pullback(Δ)
        ∇A = SparseEmbeddingUpdate(Δ, I)
        Zygote.accum_param(cx, A, ∇A)
        return (nothing, ∇A, nothing)
    end
```
The key feature here is the `Zygote.accum_param` call, which registers the update with the autodiff-context.
I really don't know why Zygote is failing to do this automatically, but there you go.

