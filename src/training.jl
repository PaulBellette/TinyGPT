
function lm_loss_and_accuracy(model, ps, st, x, y, mask)
    # x:    (T, B) integer token ids
    # y:    (T, B) integer token ids
    # mask: (T, B) Bool, true where target is valid (not pad)

    logits, st_new = model(x, ps, st)   # (V, T, B)

    V, T, B = size(logits)
    @assert size(y) == (T, B)
    @assert size(mask) == (T, B)

    total_loss = 0.0f0
    total_count = 0
    total_correct = 0

    for b in 1:B
        for t in 1:T
            if mask[t, b]
                # logits[:, t, b] is length-V
                logp = NNlib.logsoftmax(logits[:, t, b])
                total_loss -= logp[y[t, b]]
                total_count += 1

                pred = argmax(logits[:, t, b])
                total_correct += (pred == y[t, b])
            end
        end
    end

    loss = total_loss / total_count
    acc = total_correct / total_count

    return loss, acc, st_new
end

function train_step!(model, ps, st, opt_state, rng;
                     batch_size=32,
                     min_bindings=3,
                     max_bindings=5)

    batch = generate_batch(rng, batch_size;
                           min_bindings=min_bindings,
                           max_bindings=max_bindings)

    x, y = make_lm_batch(batch)
    mask = make_lm_pad_mask(y)

    loss_fn(ps_local) = begin
        loss, _, st_new = lm_loss_and_accuracy(model, ps_local, st, x, y, mask)
        return loss, st_new
    end

    (loss, st_new), grads = Zygote.withgradient(loss_fn, ps)

    opt_state, ps = Optimisers.update(opt_state, ps, grads[1])

    # recompute accuracy after update is optional; here we use pre-update acc for logging
    _, acc, _ = lm_loss_and_accuracy(model, ps, st_new, x, y, mask)

    return ps, st_new, opt_state, loss, acc
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

function train!(
    model;
    steps=2000,
    batch_size=32,
    lr=1f-3,
    seed=42,
    min_bindings=3,
    max_bindings=5,
)
    rng = MersenneTwister(seed)

    ps, st = LuxCore.setup(rng, model)
    opt = Optimisers.AdamW(lr)
    opt_state = Optimisers.setup(opt, ps)

    losses = Float32[]
    accs = Float32[]

    for step in 1:steps
        ps, st, opt_state, loss, acc = train_step!(
            model, ps, st, opt_state, rng;
            batch_size=batch_size,
            min_bindings=min_bindings,
            max_bindings=max_bindings,
        )

        push!(losses, loss)
        push!(accs, acc)

        if step % 50 == 0
            #do a test of answer only accuracy of a test batch (it is post grad but whatever)
            test_batch = generate_batch(MersenneTwister(1000000 + step), 256, min_bindings=3, max_bindings=5)
            test_acc = answer_only_accuracy(model, ps, st, test_batch)
            println("step=$(step) loss=$(round(loss, digits=4)) acc=$(round(acc, digits=4)) test_acc=$(round(test_acc, digits=4))")
        end
    end

    return ps, st, losses, accs
end