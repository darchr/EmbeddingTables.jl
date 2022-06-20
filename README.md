# EmbeddingTables

Support for Embedding Tables with sparse updates compatible with the Flux and Zygote ecosystem.

--------------

## Lookup API

Embedding table lookups consist of four main functions: `lookup`, `lookup!`, `maplookup` and `maplookup!`.
These are described below:

### `lookup`

```julia
lookup(A::AbstractEmbeddingTable, inds::Union{AbstractVector, AbstractMatrix}) -> AbstractMatrix
```
Perform embedding table lookup on table `A` using indicies `inds`.
If `inds` is an `AbstractVector`, then the result `O` is defined as
```julia
O[:, i] = A[:, inds[i]]
```
If `inds` is an `AbstractMatrix`, then the result `O` is defined as
```julia
O[:, i] = sum(i -> A[:, inds[i, j]], axes(inds, 1))
```

**Example**

```julia
julia> using EmbeddingTables

julia> data = [j for _ in 1:5, j in 1:5]
5×5 Matrix{Int64}:
 1  2  3  4  5
 1  2  3  4  5
 1  2  3  4  5
 1  2  3  4  5
 1  2  3  4  5

julia> A = SimpleEmbedding(data)
5×5 SimpleEmbedding{Dynamic, Int64, Matrix{Int64}}:
 1  2  3  4  5
 1  2  3  4  5
 1  2  3  4  5
 1  2  3  4  5
 1  2  3  4  5

julia> inds = [1,3,4,4,2,5]

# Non-summing embedding table lookup
julia> lookup(A, inds)
5×6 Matrix{Int64}:
 1  3  4  4  2  5
 1  3  4  4  2  5
 1  3  4  4  2  5
 1  3  4  4  2  5
 1  3  4  4  2  5

julia> inds = [
    1 4;
    2 5;
]

# Reducing lookup.
# Column 1 of the output is the sum of columns 1 and 2 of A.
# Column 2 of the output is the sum of columns 4 and 5 of A.
julia> lookup(A, inds)
5×2 Matrix{Int64}:
 3  9
 3  9
 3  9
 3  9
 3  9
```

### `lookup!`

```julia
lookup!(O::AbstractMatrix, A::AbstractEmbeddingTable, inds)
```

In place version of `lookup`.
That is, `lookup` can be (and is) implemented as
```julia
function lookup(A, inds)
    O = destination(A, inds)
    lookup!(O, A, inds)
    return O
end
```
where `EmbeddingTables.destination` allocates the appropriate output container.

### `maplookup`

```julia
maplookup(
    [strategy::AbstractExecutionStrategy],
    tables::AbstractVector{<:AbstractEmbeddingTable],
    indices
)
```
Perform multiple independent lookups on multiple embedding tables.
Argument `indices` can be:
* `AbstractVector{<:Union{AbstractVector{<:Integer},AbstractMatrix{<:Integer}}}`.
  In this case, `length(indices) == length(tables)` must hold and `maplookup` will
  effectively broadcast `lookup` across `tables` and `indices`.
* `AbstractMatrix{<:Integer}`: In this case, `size(indices, 2) == length(tables)` must hold
  and `lookup` will be broadcasted across `tables` and each column of `indices`, performing
  non-reducing lookups.

**Example**
```julia
julia> using EmbeddingTables

julia> A = SimpleEmbedding([j for _ in 1:2, j in 1:3])
2×3 SimpleEmbedding{Dynamic, Int64, Matrix{Int64}}:
 1  2  3
 1  2  3

julia> B = SimpleEmbedding([10j for _ in 1:2, j in 1:3])
2×3 SimpleEmbedding{Dynamic, Int64, Matrix{Int64}}:
 10  20  30
 10  20  30

julia> tables = [A, B]
2-element Vector{SimpleEmbedding{Dynamic, Int64, Matrix{Int64}}}:
 [1 2 3; 1 2 3]
 [10 20 30; 10 20 30]

julia> iA = [1,2,1]
3-element Vector{Int64}:
 1
 2
 1

julia> iB = [2,1,1]
3-element Vector{Int64}:
 2
 1
 1

julia> indices = [iA, iB]
2-element Vector{Vector{Int64}}:
 [1, 2, 1]
 [2, 1, 1]

julia> results = maplookup(tables, indices)
2-element Vector{Matrix{Int64}}:
 [1 2 1; 1 2 1]
 [20 10 10; 20 10 10]

julia> results == [lookup(A, iA), lookup(B, iB)]
true

# Can achieve similar results using a Matrix of indices
julia> combined = hcat(iA, iB)

julia> results == maplookup(tables, combined)
true
```

**Strategies**
The function `maplookup` accepts an optional `AbstractExecutionStrategy` as its first argument.
These are used to apply different parallelization strategies and possible post-op concatenation fusion for better performance.
Implemented strategies include:
* `DefaultStrategy`: Perform each lookup sequentially on a single thread. (**Default**)
* `SimpleParallelStrategy`: Statically parallelize lookups across each table using all available threads.
  Does not use intra-table parallelism (multiple threads working on a single table).
* `PreallocationStrategy`: Fuse ensemble lookup with a post-op concatenation.
  The result is a single matrix of results where:
  ```julia
  maplookup(PreallocationStrategy(), table, indices) == reduce(vcat, maplookup(table, indices))
  ```
  The `PreallocationStrategy` takes an optional integer argument `prependrows` which will insert `prependrows` before the actual embedding table lookups.
  This is helpful for workloads like DLRM that concatenate the results of the bottom dense network with the results of embedding table lookups.
  **Note**: This implementation also uses intra-table parallelism.

### `maplookup!`

The in-place version of `maplookup`.

--------------

## Back Propagation

Embedding tables support lazy back-propagation with fused stochastic gradient descent updates compatible with `Flux.jl`.
An example is shown below:
```julia
julia> using EmbeddingTables, Flux

julia> A = SimpleEmbedding(zeros(Int, 4, 4))

julia> inds = [1,3,4]

# Do reverwse mode automatic differentiation on `lookup`.
# The backwards pass is available in the closure `pullback`.
julia> y, pullback = Flux.Zygote._pullback(lookup, A, inds);

julia> y
4×3 Matrix{Float32}:
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

# Construct a dummy adjoint value with the same shape as `y`.
julia> adjoint = Float32[1 5 9; 2 6 10; 3 7 11; 4 8 12]
4×3 Matrix{Float32}:
 1.0  5.0   9.0
 2.0  6.0  10.0
 3.0  7.0  11.0
 4.0  8.0  12.0

# Grab the second element of the result of `pullback`, which
# is the gradient for the embedding table.
julia> gradient = pullback(adjoint)[2]
SparseEmbeddingUpdate{Dynamic, Matrix{Float32}, Vector{Int64}}(Float32[1.0 5.0 9.0; 2.0 6.0 10.0; 3.0 7.0 11.0; 4.0 8.0 12.0], [1, 3, 4])

# Apply stochastic gradient descent
julia> optimizer = Flux.Descent(0.1)

julia> EmbeddingTables.update!(optimizer, table, gradient); A
4×4 SimpleEmbedding{Dynamic, Float32, Matrix{Float32}}:
 -0.1  0.0  -0.5  -0.9
 -0.2  0.0  -0.6  -1.0
 -0.3  0.0  -0.7  -1.1
 -0.4  0.0  -0.8  -1.2
```

The stochastic gradient descent operation comes in two forms, a single-table form and a multi-table form (corresponding to `maplookup`).
These are described in detail below.

### Single Table `update!`
```julia
update!(
    opt::Flux.Descent,
    table::AbstractEmbeddingTable,
    gradient,
    [indexer = EmbeddingTables.Indexer()],
    [nontemporal = Val{true}()]
)
```
Perform a stochastic gradient descent update operation on `table` using the gradient `gradient`.
Optional argument `indexer` is an auxiliary helper struct which can be preallocated and passed explicitly to help with performance.
Final argument `nontemporal` is a `Val` type argument that indicates whether non-temporal stores should be used.
In some cases, this can improve performance and rarely harms performance.

### Multi-Table `update!`.

```julia
function update!(
    opt::Flux.Descent,
    tables::AbstractVector{<:AbstractEmbeddingTable},
    gradients::AbstractVector{<:SparseEmbeddingUpdate},
    indexers::AbstractVector{<:AbstractIndexer},
    [nontemporal = Val{true}()];
    kw...
)
```
This will group together multiple stochastic gradient descent updates in a parallel manner for better performance.
In this case, the auxiliary `indexers` must be preallocated and passed explicitly, which can be done using
```julia
indexers = [Indexer() for _ in tables]
```
**Keywords**

* `nthreads::Integer`: The number of threads to use for parallelization. Default: `Threads.nthreads()`.
* `num_splits::Integer`: The number of chunks to use for intra-table parallelism. Default: 4
* `scratchspaces::Vector`: Thread local scratch space for partial accumulations.
  Only needed for `Dynamic` embedding tables (described later).
  To avoid allocating these every operation, these can be preallocated using
```julia
scratchspaces = [EmbeddingTables.scratch(tables[1]) for _ in 1:Threads.nthreads()]`
```
