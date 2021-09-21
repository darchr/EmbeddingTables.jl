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

#####
##### Verifcation
#####

# Some bugs can arise if tables forget to define a custom implementation of "columnview".
# This causes a view to be wrapped around the embedding table itself which can cause
# unexpected behavior - especially if we're reyling on "columnview" to study certain behavior.
struct DummyEmbedding{S,T} <: AbstractEmbeddingTable{S,T}
    data::Matrix{T}
end

Base.size(A::DummyEmbedding) = size(A.data)
Base.@propagate_inbounds Base.getindex(A::DummyEmbedding, i::Int) = A.data[i]
Base.@propagate_inbounds Base.setindex!(A::DummyEmbedding, v, i::Int) = (A.data[i] = v)
Base.pointer(A::DummyEmbedding) = pointer(A.data)
EmbeddingTables.columnpointer(A::DummyEmbedding, i::Integer) = columnpointer(A.data, i)
EmbeddingTables.example(A::DummyEmbedding) = A.data

# Intenionally incorrect signature.
function EmbeddingTables.columnview(::DummyEmbedding)
    return -1
end

@testset "Testing `columnview` error" begin
    table = DummyEmbedding{Dynamic,Float32}(randn(Float32, 10, 10))
    inds = rand(Int, size(table, 2), 10)
    @test_throws ArgumentError EmbeddingTables.lookup(table, inds)
end
