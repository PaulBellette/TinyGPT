function lm_loss(model, ps, st, d)
    x, y, mask = d
    logits, st = model(x, ps, st)   # (V, T, B)

    V, T, B = size(logits)
    N = T * B

    logits2 = reshape(logits, V, N)
    y2 = reshape(y, N)
    mask2 = reshape(mask, N)

    logp = NNlib.logsoftmax(logits2; dims=1)

    # one-hot targets: (V, N)
    oh = onehotbatch(y2, 1:V)

    # pick gold logprobs by reduction instead of scalar indexing
    gold_logp = sum(logp .* oh; dims=1)   # (1, N)
    gold_logp = vec(gold_logp)            # (N,)

    w = Float32.(mask2)
    loss = -sum(gold_logp .* w) / sum(w)

    return loss, st, (;)
end

function answer_only_accuracy(model, ps, st, batch)
    x, y = make_lm_batch(batch)
    logits, _ = model(x, ps, st)   # (V, T, B)

    T, B = size(y)
    pad_id = TOK2ID["<PAD>"]

    correct = 0
    total = 0

    for b in 1:B
        # find last non-pad target position
        t_ans = findlast(t -> y[t, b] != pad_id, 1:T)
        if t_ans !== nothing
            pred = argmax(logits[:, t_ans, b])
            correct += (pred == y[t_ans, b])
            total += 1
        end
    end

    return correct / total
end

function get_device()
    return CUDA.functional() ? gpu_device() : cpu_device()
end

function train!(
    model;
    steps=2000,
    batch_size=32,
    lr=1f-3,
    seed=42,
    min_bindings=3,
    max_bindings=5,
    backend = "cpu",
    validation_interval = 50,
    early_stop = false,
    early_stop_threshold = 0.95
)

    Reactant.set_default_backend(backend)
    dev = reactant_device()
    rng = MersenneTwister(seed)

    ps, st = dev(LuxCore.setup(rng, model))
    opt_state = Training.TrainState(model, ps, st, Optimisers.AdamW(lr))
    losses = Float32[]
    test_accuracy = Float32[]
    batch_timings = []
    start_time_interval = time_ns()
    for step in 1:steps
        batch = generate_batch(rng, batch_size;
                           min_bindings=min_bindings,
                           max_bindings=max_bindings)
        x, y = make_lm_batch(batch)
        mask = make_lm_pad_mask(y)
        (_, loss, _, opt_state) = Training.single_train_step!(
            AutoEnzyme(),
            lm_loss,
            (dev(x), dev(y), dev(Array(mask))),
            opt_state
        )

        loss = cpu_device()(loss)
        push!(losses, loss)

        if step % validation_interval == 0
            stop_time_interval = time_ns()
            time_per_batch_ms = ((stop_time_interval - start_time_interval) / validation_interval) / 1e6
            push!(batch_timings, time_per_batch_ms)
            test_batch = generate_batch(
                MersenneTwister(1000000 + step), 
                256, 
                min_bindings=3, 
                max_bindings=5
            )
            test_acc = answer_only_accuracy(
                model, 
                cpu_device()(opt_state.parameters), 
                cpu_device()(Lux.testmode(opt_state.states)), 
                test_batch
            )
            push!(test_accuracy, test_acc)
            @info("step=$(step), training_loss=$(round(loss, digits=4)), test_acc=$(round(test_acc, digits = 4)), time_per_batch = $(round(time_per_batch_ms, digits = 4)) ms")
            start_time_interval = time_ns()
            if early_stop
                if test_acc > early_stop_threshold
                    break
                end
            end
        end
    end

    return cpu_device()(opt_state.parameters), cpu_device()(opt_state.states), losses, test_accuracy, batch_timings
end