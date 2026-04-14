
const KEYS   = ["A", "B", "C", "D", "E", "F"]
const VALUES = ["0", "1", "2", "3", "4", "5"]
const PUNCT  = [":", ";", "?", "=", "<PAD>"]

const VOCAB = vcat(KEYS, VALUES, PUNCT)

const TOK2ID = Dict(tok => i for (i, tok) in enumerate(VOCAB))
const ID2TOK = Dict(i => tok for (i, tok) in enumerate(VOCAB))

tokenize(tokens::Vector{String}) = [TOK2ID[t] for t in tokens]
detokenize(ids::Vector{Int}) = [ID2TOK[i] for i in ids]

function decode_tokens(ids::Vector{Int})
    join(detokenize(ids), " ")
end

function generate_example(
    rng::AbstractRNG;
    min_bindings::Int = 3,
    max_bindings::Int = 5,
)
    n_bindings = rand(rng, min_bindings:max_bindings)

    chosen_keys = randperm(rng, length(KEYS))[1:n_bindings]
    chosen_keys = KEYS[chosen_keys]

    assigned_vals = [rand(rng, VALUES) for _ in 1:n_bindings]

    tokens = String[]

    for i in 1:n_bindings
        push!(tokens, chosen_keys[i])
        push!(tokens, ":")
        push!(tokens, assigned_vals[i])
        push!(tokens, ";")
    end

    q_idx = rand(rng, 1:n_bindings)
    q_key = chosen_keys[q_idx]
    q_val = assigned_vals[q_idx]

    push!(tokens, "?")
    push!(tokens, q_key)
    push!(tokens, "=")
    push!(tokens, q_val)

    return tokenize(tokens)
end

function generate_batch(
    rng::AbstractRNG,
    batch_size::Int;
    min_bindings::Int = 3,
    max_bindings::Int = 5,
)
    examples = [
        generate_example(rng; min_bindings=min_bindings, max_bindings=max_bindings)
        for _ in 1:batch_size
    ]

    max_len = maximum(length.(examples))
    pad_id = TOK2ID["<PAD>"]

    batch = fill(pad_id, max_len, batch_size)

    for (j, ex) in enumerate(examples)
        batch[1:length(ex), j] .= ex
    end

    return batch
end

function make_lm_batch(batch::Matrix{Int})
    x = batch[1:end-1, :]
    y = batch[2:end, :]
    return x, y
end

function make_lm_pad_mask(y::Matrix{Int})
    pad_id = TOK2ID["<PAD>"]
    y .!= pad_id
end