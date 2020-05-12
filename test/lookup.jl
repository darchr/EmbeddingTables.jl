function test_routine(baseline, new)
    @test size(baseline) == size(new)
    @test length(baseline) == length(new)

    nrows, ncols = size(new)

    # Lookup all indices
    I = shuffle(1:ncols)

    lookup_baseline = EmbeddingTables.lookup(baseline, I)
    lookup_new = EmbeddingTables.lookup(new, I)

    equal = (lookup_baseline == lookup_new)
    @test equal
    if !equal
        # Find all the mismatching columns.
        mismatch_cols = findall(eachcol(lookup_baseline) .!= eachcol(lookup_new))
        @show mismatch_cols
    end

    # Allow repeates
    I = fill(rand(1:ncols), 10)
    lookup_baseline = EmbeddingTables.lookup(baseline, I)
    lookup_new = EmbeddingTables.lookup(new, I)
    @test lookup_baseline == lookup_new
end

@testset "Testing Lookup" begin
    nrows = 32
    ncols = 1000

    base = rand(Float32, nrows, ncols)

    @testset "Testing Simple" begin
        A = copy(base)
        B = EmbeddingTables.SimpleEmbedding(copy(base))
        test_routine(A, B)
    end

    @testset "Testing Split" begin
        chunk_sizes = [10, 20, 30, 40, 50]

        for cols_per_chunk in chunk_sizes
            A = copy(base)
            B = EmbeddingTables.SplitEmbedding(copy(base), cols_per_chunk)
            test_routine(A, B)
        end
    end
end
