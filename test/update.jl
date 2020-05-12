function update_routine(baseline, new, iters)
    @test size(baseline) == size(new)
    @test length(baseline) == length(new)

    nrows, ncols = size(new)

    # Create a loss function and loss input
    loss(A, I, x) = Flux.mse(lookup(A, I), x)

    opt = Flux.Descent(0.1)
    for iter in 1:iters
        # Create lookup indices and an example loss.
        #
        # Lookup a random number of indices as well
        nindices = rand(div(ncols,2):ncols)
        I = rand(1:ncols, nindices)

        loss_input = randn(Float32, nrows, nindices)

        grads_baseline = Zygote.gradient(Params((baseline,)))  do
            loss(baseline, I, loss_input)
        end
        Flux.Optimise.update!(opt, baseline, grads_baseline.grads[baseline])

        grads_new = Zygote.gradient(Params((new,)))  do
            loss(new, I, loss_input)
        end
        Flux.Optimise.update!(opt, new, grads_new.grads[new])

        # Update should affect both the `baseline` and the `new` table the same.
        equal = isapprox(baseline, new)
        @test equal
        if !equal
            # Find all the mismatching columns.
            mismatch_cols = findall(eachcol(baseline) .!= eachcol(new))
            @show mismatch_cols
            printstyled(stdout, "Baseline\n"; color = :cyan)
            display(baseline)
            printstyled(stdout, "New\n"; color = :cyan)
            display(new)
            printstyled(stdout, "Difference\n"; color = :cyan)
            display(!isapprox.(new, baseline))
            println()
        end
    end
end

@testset "Testing Update" begin
    nrows = 32
    ncols = 1000

    base = randn(Float32, nrows, ncols)

    @testset "Simple" begin
        A = copy(base)
        B = EmbeddingTables.SimpleEmbedding(copy(base))
        update_routine(A, B, 100)
    end

    @testset "Split" begin
        chunk_sizes = [10, 20, 30, 40, 50]

        for cols_per_chunk in chunk_sizes
            A = copy(base)
            B = EmbeddingTables.SplitEmbedding(copy(base), cols_per_chunk)
            update_routine(A, B, 100)
        end
    end
end
