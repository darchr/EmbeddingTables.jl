@testset "Testing misc utils" begin
    @testset "`maybe` functions" begin
        x = [1, 2, 3, 4]
        y = [
            1 2
            3 4
        ]

        @test collect(EmbeddingTables.columns(x)) == [(1, 1), (2, 2), (3, 3), (4, 4)]
        @test collect(EmbeddingTables.columns(y)) == [(1, 1), (1, 3), (2, 2), (2, 4)]
    end

    @testset "Testing `OrderCount` and `ColOffset`" begin
        x = EmbeddingTables.OrderCount(10, 20)
        a, b = x    # indexed_iterate
        @test a == 10
        @test b == 20
        a, b = zero(typeof(x))
        @test iszero(a)
        @test iszero(b)
        @test propertynames(x) == (:order, :count)

        x = EmbeddingTables.ColOffset(10, 20)
        a, b = x    # indexed_iterate
        @test a == 10
        @test b == 20
        a, b = zero(typeof(x))
        @test iszero(a)
        @test iszero(b)
        @test propertynames(x) == (:col, :offset)
    end

    @testset "Testing histogram" begin
        OrderCount = EmbeddingTables.OrderCount

        dict = Dictionaries.Dictionary{Int,EmbeddingTables.OrderCount{Int}}()
        array = Vector{EmbeddingTables.OrderCount{Int}}()

        A = reduce(vcat, (fill(2, 10), fill(1, 5), fill(20, 3), 5))
        EmbeddingTables.resize_histogram!(dict, 20)
        EmbeddingTables.resize_histogram!(array, 20)

        EmbeddingTables.histogram!(dict, A)
        EmbeddingTables.histogram!(array, A)

        @test dict[2] == OrderCount(1, 10)
        @test dict[1] == OrderCount(2, 5)
        @test dict[20] == OrderCount(3, 3)
        @test dict[5] == OrderCount(4, 1)
        @test collect(keys(dict)) == [2, 1, 20, 5]

        @test array[2] == OrderCount(1, 10)
        @test array[1] == OrderCount(2, 5)
        @test array[20] == OrderCount(3, 3)
        @test array[5] == OrderCount(4, 1)

        # Run again - make sure we get the same results
        # (i.e., out `shallow_empty!` implementation works)
        EmbeddingTables.histogram!(dict, A)
        EmbeddingTables.histogram!(array, A)

        @test dict[2] == OrderCount(1, 10)
        @test dict[1] == OrderCount(2, 5)
        @test dict[20] == OrderCount(3, 3)
        @test dict[5] == OrderCount(4, 1)
        @test collect(keys(dict)) == [2, 1, 20, 5]

        @test array[2] == OrderCount(1, 10)
        @test array[1] == OrderCount(2, 5)
        @test array[20] == OrderCount(3, 3)
        @test array[5] == OrderCount(4, 1)
    end

    @testset "Testing Indexer" begin
        ColOffset = EmbeddingTables.ColOffset
        indexers = [EmbeddingTables.SparseIndexer(), EmbeddingTables.DenseIndexer()]

        for indexer in indexers
            A = [10, 4, 10, 100, 4, 4, 4, 1, 9, 10, 5]
            for _ = 1:2
                EmbeddingTables.index!(indexer, A, maximum(A))

                # Should have accounted for all entries in the histogram dict.
                @test all(x -> iszero(x.count), values(indexer.histogram))
                expected_cumulative = [
                    ColOffset(10, 1),
                    ColOffset(4, 1 + count(isequal(10), A)),
                    ColOffset(100, 1 + count(in((10, 4)), A)),
                    ColOffset(1, 1 + count(in((10, 4, 100)), A)),
                    ColOffset(9, 1 + count(in((10, 4, 100, 1)), A)),
                    ColOffset(5, 1 + count(in((10, 4, 100, 1, 9)), A)),
                    ColOffset(0, 1 + length(A)),
                ]
                @test indexer.cumulative == expected_cumulative

                #! format: off
                expected_map = [
                    #=  10 =# 1, 3, 10,
                    #=   4 =# 2, 5, 6, 7,
                    #= 100 =# 4,
                    #=   1 =# 8,
                    #=   9 =# 9,
                    #=   5 =# 11,
                ]
                #! format: on

                @test expected_map == indexer.map
            end
        end
    end
end
