#####
##### Testing `maplookup` and its various flavors.
#####

@testset "Testing `maplookup`" begin
    # We need to test "maplookup" for when we supply both vectors of arrays and higher
    # dimensional arrays.
    #
    # For example, non-reducing ensemble lookups can be performed using either a vector
    # of vectors, or a single matrix.
    #
    # Reducing ensemble lookups and be performed with a vector of matrices or a 3d array.

    ncols = 100
    _nrows = [16, 64, 512]
    ntables = 10
    ntests = 100
    nlookups = 10
    batchsize = 64

    make_base(nrows) = [randn(Float32, nrows, ncols) for _ in 1:ntables]
    function make_tables(v::Vector{<:AbstractMatrix}; static = true)
        return map(v) do matrix
            if static
                return SimpleEmbedding{Static{size(matrix,1)}}(matrix)
            else
                return SimpleEmbedding{Dynamic}(matrix)
            end
        end
    end

    @testset "Testing Non-Reducing" begin
        for _ in Base.OneTo(ntests), nrows in _nrows
            base = make_base(nrows)

            ## Vector of Vectors
            inds = [rand(1:ncols, batchsize) for _ in 1:ntables]
            reference = reduce(vcat, map(lookup, base, inds))
            tables = make_tables(base; static = true)

            # Try the three strategies
            out_default = maplookup(DefaultStrategy(), tables, inds)
            @test reduce(vcat, out_default) == reference

            out_parallel = maplookup(SimpleParallelStrategy(), tables, inds)
            @test reduce(vcat, out_parallel) == reference

            out_preallocated = maplookup(PreallocationStrategy(), tables, inds)
            @test out_preallocated == reference

            ## Matrix
            inds = rand(1:ncols, batchsize, ntables)
            reference = reduce(vcat, map(lookup, base, eachcol(inds)))
            tables = make_tables(base; static = true)

            # Try the three strategies
            out_default = maplookup(DefaultStrategy(), tables, inds)
            @test reduce(vcat, out_default) == reference

            out_parallel = maplookup(SimpleParallelStrategy(), tables, inds)
            @test reduce(vcat, out_parallel) == reference

            out_preallocated = maplookup(PreallocationStrategy(), tables, inds)
            @test out_preallocated == reference
        end
    end

    @testset "Testing Reducing" begin
        for _ in Base.OneTo(ntests), nrows in _nrows
            base = make_base(nrows)

            ## Vector of Matrices
            inds = [rand(1:ncols, nlookups, batchsize) for _ in 1:ntables]
            reference = reduce(vcat, map(lookup, base, inds))
            tables = make_tables(base; static = true)

            # Try the three strategies
            out_default = maplookup(DefaultStrategy(), tables, inds)
            @test reduce(vcat, out_default) == reference

            out_parallel = maplookup(SimpleParallelStrategy(), tables, inds)
            @test reduce(vcat, out_parallel) == reference

            out_preallocated = maplookup(PreallocationStrategy(), tables, inds)
            @test out_preallocated == reference

            ## 3D Array
            inds = rand(1:ncols, nlookups, batchsize, ntables)
            reference = reduce(vcat, map(lookup, base, eachslice(inds; dims = 3)))
            tables = make_tables(base; static = true)

            # Try the three strategies
            out_default = maplookup(DefaultStrategy(), tables, inds)
            @test reduce(vcat, out_default) == reference

            out_parallel = maplookup(SimpleParallelStrategy(), tables, inds)
            @test reduce(vcat, out_parallel) == reference

            out_preallocated = maplookup(PreallocationStrategy(), tables, inds)
            @test out_preallocated == reference
        end
    end
end

# We have a problem where lookups mapped across a vector of Embedding Tables doesn't
# correctly register the gradients.
#
# This test checks both that this case still fails and tests my workaround.
function create_ensemble(dims)
    return [EmbeddingTables.SimpleEmbedding(rand(Float32, dim)) for dim in dims]
end

_normalize(x::AbstractVector) = reduce(vcat, x)
_normalize(x) = x

loss_wrap(f) = (y, x...) -> Flux.mse(_normalize(f(x...)), y)

@testset "Testing Map" begin
    # Create three lookup tables
    dims = [(5, 5), (5, 10), (5, 15)]

    # Create input indices and the "expected" result.
    batchsize = 5
    I = [rand(1:last(d), batchsize) for d in dims]
    y = rand(Float32, sum(first.(dims)), batchsize)

    # Create the function over normal arrays to avoid hitting our workaround.
    tables = create_ensemble(dims)
    grads = Zygote.gradient(loss_wrap((x...) -> map(lookup, x...)), y, tables, I)

    # Gradients for the lookup table are at index 2
    for (i, grad) in enumerate(grads[2])
        @test isa(grad, EmbeddingTables.SparseEmbeddingUpdate)
        # Make sure indices were captured correctly.
        @test grad.indices == I[i]
    end

    #####
    ##### Now, verify that `SimpleEmbedding` works.
    #####

    grads = Zygote.gradient(loss_wrap(maplookup), y, tables, I)
    for (i, grad) in enumerate(grads[2])
        @test isa(grad, EmbeddingTables.SparseEmbeddingUpdate)
        # Make sure indices were captured correctly.
        @test grad.indices == I[i]
    end

    #####
    ##### Try "PreallocationStrategy"
    #####

    f = (x...) -> maplookup(PreallocationStrategy(), x...)
    grads2 = Zygote.gradient(loss_wrap(f), y, tables, I)

    for (i, grad) in enumerate(grads2[2])
        @test isa(grad, EmbeddingTables.SparseEmbeddingUpdate)
        # Make sure indices were captured correctly.
        @test grad.indices == I[i]
        @test grad.delta == grads[2][i].delta
        @test grad.indices == grads[2][i].indices
    end

    # Retry with prepadding
    # Use a view of the output to keep dimensions happy with the "y" matrix used as the
    # dummy target.
    f = (x...) -> @views(maplookup(PreallocationStrategy(20), x...)[21:end, :])
    grads2 = Zygote.gradient(loss_wrap(f), y, tables, I)

    for (i, grad) in enumerate(grads2[2])
        @test isa(grad, EmbeddingTables.SparseEmbeddingUpdate)
        # Make sure indices were captured correctly.
        @test grad.indices == I[i]
        @test grad.delta == grads[2][i].delta
        @test grad.indices == grads[2][i].indices
    end
end
