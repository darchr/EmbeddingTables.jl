using EmbeddingTables
using Test

# For testing gradients and SGD
using Flux
using Zygote

include("map.jl")

@testset "EmbeddingTables.jl" begin
    # First, just test that an embedding table works like an array.
    data = Float32.([
        1 2 3;
        4 5 6;
    ])

    data_copy = copy(data)

    A = SimpleEmbedding(data)
    I = [3,1]
    expected = [
        3 1;
        6 4;
    ]

    @test lookup(A, [3,1]) == expected
    @test size(A) == size(data)

    loss_input = [
        2.5 0.5;
        5.5 4.5;
    ]

    # Now, test some simple Zygote gradients.
    #params = Params((A,))
    grads = Zygote.gradient(Params((A,))) do
        Flux.mse(lookup(A, I), loss_input)
    end

    # Apply to a known forward
    grads_test = Zygote.gradient(Params((data_copy,))) do
        Flux.mse(data_copy[:, I], loss_input)
    end

    @test haskey(grads.grads, A)

    # Apply the update
    opt = Flux.Descent(1.0)
    Flux.Optimise.update!(opt, A, grads.grads[A])
    Flux.Optimise.update!(opt, data_copy, grads_test.grads[data_copy])

    @test A == data_copy
end
