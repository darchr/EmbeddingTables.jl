non_reducing_update(x...; kw...) = _update_inner(i -> (i,), x...; kw...)
reducing_update(x...; kw...) = _update_inner(i -> (i, i), x...; kw...)

function _update_inner(
    f::F,
    table::AbstractEmbeddingTable,
    baseline::Matrix;
    numtests = 10,
) where {F}
    @test size(table) == size(baseline)
    @test length(table) == length(table)
    ncols = size(table, 2)

    opt = Flux.Descent(10.0)
    for _ = 1:numtests
        # Generate random lookup indices which may include repeats.
        indices_base = rand(1:ncols, f(ncols)...)
        indices = copy(indices_base)

        out_ref, back_ref = Zygote._pullback(lookup, baseline, indices)
        out, back = Zygote._pullback(lookup, table, indices)

        @test out == out_ref

        # Seed for the sensitivity.
        diff_out = randn(Float32, size(out))

        # The results here have different types, so we can't compare them directly.
        # Instead, we need to use the `uncompress` function to turn the `diff_table`
        # into a full array.
        diff_baseline = back_ref(diff_out)
        @test length(diff_baseline) == 3
        @test diff_baseline[1] === nothing
        @test diff_baseline[3] === nothing
        diff_baseline = diff_baseline[2]

        diff_table = back(diff_out)
        @test length(diff_table) == 3
        @test diff_table[1] === nothing
        @test diff_table[3] === nothing
        diff_table = diff_table[2]

        @test isa(diff_table, SparseEmbeddingUpdate)
        uncompressed = EmbeddingTables.uncompress(diff_table, size(diff_baseline, 2))
        @test isapprox(diff_baseline, uncompressed)

        # Need to create a new closure because "crunch" effectively destroys the old
        # version of "indices" that gets captured by the original pullback.
        out_ref, back_ref = Zygote._pullback(lookup, baseline, copy(indices_base))
        out, back = Zygote._pullback(lookup, table, copy(indices_base))

        diff_baseline = back_ref(diff_out)[2]
        diff_table = back(diff_out)[2]

        zeros_baseline = zeros(eltype(baseline), size(baseline))
        zeros_table = zeros(table)

        @test isa(zeros_table, AbstractEmbeddingTable)
        Flux.Optimise.update!(opt, zeros_baseline, diff_baseline)
        Flux.Optimise.update!(opt, zeros_table, diff_table)
        @test isapprox(zeros_baseline, zeros_table)

        # Also try with with the partitioner to ensure that it's logic is correct.
        out_ref, back_ref = Zygote._pullback(lookup, copy(baseline), copy(indices_base))
        out, back = Zygote._pullback(lookup, table, copy(indices_base))

        diff_baseline = back_ref(diff_out)[2]
        diff_table = back(diff_out)[2]

        zeros_baseline = zeros(eltype(baseline), size(baseline))
        zeros_table = zeros(table)

        @test isa(zeros_table, AbstractEmbeddingTable)
        Flux.Optimise.update!(opt, zeros_baseline, diff_baseline)
    end
end

#####
##### Tests
#####

@testset "Testing Update Partitions" begin
    batchsize = 512
    base = randn(Float32, 16, 100)
    featuresize = size(base, 1)
    A = SimpleEmbedding{Static{featuresize}}(copy(base))
    delta = randn(Float32, featuresize, batchsize)
    inds = rand(1:size(base, 2), batchsize)

    grad = EmbeddingTables.SparseEmbeddingUpdate{Static{featuresize}}(delta, inds)
    indexer = EmbeddingTables.Indexer()
    EmbeddingTables.index!(indexer, grad.indices)

    # Do the reference update
    EmbeddingTables.update!(A, grad, indexer, Float32(1.0))

    # Now, do a partitioned update.
    B = SimpleEmbedding{Static{featuresize}}(copy(base))
    num_splits = 4
    for this_split in Base.OneTo(num_splits)
        EmbeddingTables.update!(
            B,
            grad,
            EmbeddingTables.IndexerView(indexer, num_splits, this_split),
            Float32(1.0),
        )
    end
    @test A == B
end

@testset "Testing Update" begin
    nrows = [64, 80, 512]
    ncols = 100
    numtests = 10

    @testset "Simple Nonreducing" begin
        for rows in nrows
            println("Static Nonreducing: ", rows)
            base = randn(Float32, rows, ncols)
            A = SimpleEmbedding{Static{rows}}(copy(base))
            B = copy(base)
            @time non_reducing_update(A, B; numtests = numtests)
        end

        for rows in nrows
            println("Dynamic Nonreducing: ", rows)
            base = randn(Float32, rows, ncols)
            A = SimpleEmbedding{Dynamic}(copy(base))
            B = copy(base)
            @time non_reducing_update(A, B; numtests = numtests)
        end
    end

    @testset "Simple Reducing" begin
        for rows in nrows
            println("Static Reducing: ", rows)
            base = randn(Float32, rows, ncols)
            A = SimpleEmbedding{Static{rows}}(copy(base))
            B = copy(base)
            @time reducing_update(A, B; numtests = numtests)
        end

        for rows in nrows
            println("Dynamic Reducing: ", rows)
            base = randn(Float32, rows, ncols)
            A = SimpleEmbedding{Dynamic}(copy(base))
            B = copy(base)
            @time reducing_update(A, B; numtests = numtests)
        end
    end
end
