@testset "Testing Embeddingtable Constructors" begin
    # Make sure we get error for invalid constructors.

    # Feature size does not evenly distribute across cache lines.
    odd_size = rand(Float32, 65, 10)
    even_size = rand(Float32, 64, 10)

    # Make sure we have success
    x = SimpleEmbedding{Static{64}}(even_size)
    @test size(x) == size(even_size)

    # Paramter mismatch
    @test_throws ArgumentError SimpleEmbedding{Static{32}}(even_size)
    # Wrong type
    @test_throws ArgumentError SimpleEmbedding{Static{64.0}}(even_size)

    # Creating dynamic tables should pretty much always work.
    x = SimpleEmbedding(odd_size)
    @test size(x) == size(odd_size)
    @test isa(x, SimpleEmbedding{EmbeddingTables.Dynamic})

    x = SimpleEmbedding{Static{size(odd_size, 1)}}(odd_size)
    @test size(x) == size(odd_size)
    @test isa(x, SimpleEmbedding{EmbeddingTables.Static{65}})
end
