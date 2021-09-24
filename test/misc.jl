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

    @testset "Testing histogram" begin
        dict = Dictionaries.Dictionary{Int,NamedTuple{(:order, :count),Tuple{Int,Int}}}()
        A = reduce(vcat, (fill(2, 10), fill(1, 5), fill(20, 3), 5))
        EmbeddingTables.histogram!(dict, A)

        @test dict[2] == (; order = 1, count = 10)
        @test dict[1] == (; order = 2, count = 5)
        @test dict[20] == (; order = 3, count = 3)
        @test dict[5] == (; order = 4, count = 1)
        @test collect(keys(dict)) == [2, 1, 20, 5]

        # Run again - make sure we get the same results
        # (i.e., out `shallow_empty!` implementation works)
        EmbeddingTables.histogram!(dict, A)

        @test dict[2] == (; order = 1, count = 10)
        @test dict[1] == (; order = 2, count = 5)
        @test dict[20] == (; order = 3, count = 3)
        @test dict[5] == (; order = 4, count = 1)
        @test collect(keys(dict)) == [2, 1, 20, 5]
    end

    @testset "Testing Indexer" begin
        indexer = EmbeddingTables.Indexer()
        A = [10, 4, 10, 100, 4, 4, 4, 1, 9, 10, 5]
        for _ = 1:2
            EmbeddingTables.index!(indexer, A)

            # Should have accounted for all entries in the histogram dict.
            @test all(x -> iszero(x.count), values(indexer.histogram))
            expected_cumulative = [
                (col = 10, offset = 1),
                (col = 4, offset = 1 + count(isequal(10), A)),
                (col = 100, offset = 1 + count(in((10, 4)), A)),
                (col = 1, offset = 1 + count(in((10, 4, 100)), A)),
                (col = 9, offset = 1 + count(in((10, 4, 100, 1)), A)),
                (col = 5, offset = 1 + count(in((10, 4, 100, 1, 9)), A)),
                (col = 0, offset = 1 + length(A)),
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
