
struct TinyGPTConfig
    vocab_size::Int
    seq_len::Int
    d_model::Int
    n_heads::Int
    d_ff::Int
    n_layers::Int
    dropout::Float32
end

struct TinyRecycledGPTConfig
    vocab_size::Int
    seq_len::Int
    d_model::Int
    n_heads::Int
    d_ff::Int
    n_in_layers::Int
    n_middle_loops::Int
    n_middle_layers::Int
    n_out_layers::Int
    dropout::Float32
end

function CausalSelfAttention(d_model::Int, n_heads::Int; pdrop=0.0f0)
    @assert d_model % n_heads == 0
    d_head = d_model ÷ n_heads

    @compact(
        wq = Dense(d_model => d_model),
        wk = Dense(d_model => d_model),
        wv = Dense(d_model => d_model),
        wo = Dense(d_model => d_model),
        dropout = Dropout(pdrop),
    ) do x
        T, B = size(x, 2), size(x, 3)

        q = wq(x)
        k = wk(x)
        v = wv(x)

        q = reshape(q, d_head, n_heads, T, B)
        k = reshape(k, d_head, n_heads, T, B)
        v = reshape(v, d_head, n_heads, T, B)

        y, attn = LuxLib.scaled_dot_product_attention(q, k, v; is_causal=true)
        y = reshape(y, d_model, T, B)

        y = wo(y)
        y = dropout(y)

        @return y
    end
end

function make_mlp(d_model, d_ff, pdrop)
    Chain(
        Dense(d_model => d_ff, gelu),
        Dense(d_ff => d_model),
        Dropout(pdrop),
    )
end

function DecoderBlock(d_model, n_heads, d_ff; pdrop=0.0f0)
    @compact(
        ln1 = LayerNorm((d_model, 1); dims=1),
        attn = CausalSelfAttention(d_model, n_heads; pdrop),
        ln2 = LayerNorm((d_model, 1); dims=1),
        mlp = make_mlp(d_model, d_ff, pdrop),
    ) do x
        h1 = ln1(x)
        a  = attn(h1)
        x  = x .+ a

        h2 = ln2(x)
        f  = mlp(h2)
        x  = x .+ f

        @return x
    end
end

#this is a weird idea I had. You can think of a standard decode block as something like Lie-Trotter splitting
#so why not extend the metaphor and try for Strang splitting to control errors better via the symmetric update rule
function DecoderBlockStrang(d_model, n_heads, d_ff; pdrop=0.0f0)
    @compact(
        ln1 = LayerNorm((d_model, 1); dims=1),
        attn = CausalSelfAttention(d_model, n_heads; pdrop),
        ln2 = LayerNorm((d_model, 1); dims=1),
        mlp = make_mlp(d_model, d_ff, pdrop),
    ) do x
        x = x .+ 0.5f0 .* attn(ln1(x))
        x = x .+ mlp(ln2(x))
        x = x .+ 0.5f0 .* attn(ln1(x))
        @return x
    end
end

function TinyGPTModel(cfg::TinyGPTConfig)
    blocks = [DecoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff; pdrop=cfg.dropout)
              for _ in 1:cfg.n_layers]

    # blocks = [DecoderBlockStrang(cfg.d_model, cfg.n_heads, cfg.d_ff; pdrop=cfg.dropout)
        #   for _ in 1:cfg.n_layers]
    @compact(
        tok_embed  = Embedding(cfg.vocab_size => cfg.d_model),
        pos_embed  = Embedding(cfg.seq_len => cfg.d_model),
        blocks     = blocks,
        final_norm = LayerNorm((cfg.d_model, 1); dims=1),
        head       = Dense(cfg.d_model => cfg.vocab_size),
    ) do tokens
        # tokens: (T, B)
        T, B = size(tokens)

        x = tok_embed(tokens)   # (d_model, T, B)

        # 1-based learned positions for Julia
        pos_ids = reshape(collect(1:T), T, 1)
        pos_ids = repeat(pos_ids, 1, B)   # (T, B)

        p = pos_embed(pos_ids)  # (d_model, T, B)
        x = x .+ p

        for block in blocks
            x = block(x)
        end

        x = final_norm(x)
        logits = head(x)        # (vocab_size, T, B)

        @return logits
    end
end


function TinySymmetricGPTModel(cfg::TinyGPTConfig)
    blocks = [DecoderBlockStrang(cfg.d_model, cfg.n_heads, cfg.d_ff; pdrop=cfg.dropout)
          for _ in 1:cfg.n_layers]
    @compact(
        tok_embed  = Embedding(cfg.vocab_size => cfg.d_model),
        pos_embed  = Embedding(cfg.seq_len => cfg.d_model),
        blocks     = blocks,
        final_norm = LayerNorm((cfg.d_model, 1); dims=1),
        head       = Dense(cfg.d_model => cfg.vocab_size),
    ) do tokens
        # tokens: (T, B)
        T, B = size(tokens)

        x = tok_embed(tokens)   # (d_model, T, B)

        # 1-based learned positions for Julia
        pos_ids = reshape(collect(1:T), T, 1)
        pos_ids = repeat(pos_ids, 1, B)   # (T, B)

        p = pos_embed(pos_ids)  # (d_model, T, B)
        x = x .+ p

        for block in blocks
            x = block(x)
        end

        x = final_norm(x)
        logits = head(x)        # (vocab_size, T, B)

        @return logits
    end
end

function TinyRecycledGPTModel(cfg::TinyRecycledGPTConfig)
    in_blocks = [DecoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff; pdrop=cfg.dropout)
          for _ in 1:cfg.n_in_layers]
    middle_blocks = [DecoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff; pdrop=cfg.dropout) for _ in 1:cfg.n_middle_layers]
    out_blocks = [DecoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff; pdrop=cfg.dropout)
          for _ in 1:cfg.n_out_layers]
    @compact(
        tok_embed = Embedding(cfg.vocab_size => cfg.d_model),
        pos_embed = Embedding(cfg.seq_len => cfg.d_model),
        in_blocks = in_blocks,
        middle_blocks = middle_blocks,
        out_blocks = out_blocks,
        n_middle_loops = cfg.n_middle_loops,
        final_norm = LayerNorm((cfg.d_model, 1); dims=1),
        head = Dense(cfg.d_model => cfg.vocab_size),
    ) do tokens
        # tokens: (T, B)
        T, B = size(tokens)

        x = tok_embed(tokens)   # (d_model, T, B)

        # 1-based learned positions for Julia
        pos_ids = reshape(collect(1:T), T, 1)
        pos_ids = repeat(pos_ids, 1, B)   # (T, B)

        p = pos_embed(pos_ids)  # (d_model, T, B)
        x = x .+ p

        for block in in_blocks
            x = block(x)
        end

        for _ in 1 : n_middle_loops
            for block in middle_blocks
                x = block(x)
            end
        end

        for block in out_blocks
            x = block(x)
        end

        x = final_norm(x)
        logits = head(x)        # (vocab_size, T, B)

        @return logits
    end
end

function TinyRecycledSymmetricGPTModel(cfg::TinyRecycledGPTConfig)
    in_blocks = [DecoderBlockStrang(cfg.d_model, cfg.n_heads, cfg.d_ff; pdrop=cfg.dropout)
          for _ in 1:cfg.n_in_layers]
    middle_blocks = [DecoderBlockStrang(cfg.d_model, cfg.n_heads, cfg.d_ff; pdrop=cfg.dropout) for _ in 1:cfg.n_middle_layers]
    out_blocks = [DecoderBlockStrang(cfg.d_model, cfg.n_heads, cfg.d_ff; pdrop=cfg.dropout)
          for _ in 1:cfg.n_out_layers]
    @compact(
        tok_embed = Embedding(cfg.vocab_size => cfg.d_model),
        pos_embed = Embedding(cfg.seq_len => cfg.d_model),
        in_blocks = in_blocks,
        middle_blocks = middle_blocks,
        out_blocks = out_blocks,
        n_middle_loops = cfg.n_middle_loops,
        final_norm = LayerNorm((cfg.d_model, 1); dims=1),
        head = Dense(cfg.d_model => cfg.vocab_size),
    ) do tokens
        # tokens: (T, B)
        T, B = size(tokens)

        x = tok_embed(tokens)   # (d_model, T, B)

        # 1-based learned positions for Julia
        pos_ids = reshape(collect(1:T), T, 1)
        pos_ids = repeat(pos_ids, 1, B)   # (T, B)

        p = pos_embed(pos_ids)  # (d_model, T, B)
        x = x .+ p

        for block in in_blocks
            x = block(x)
        end

        for _ in 1 : n_middle_loops
            for block in middle_blocks
                x = block(x)
            end
        end

        for block in out_blocks
            x = block(x)
        end

        x = final_norm(x)
        logits = head(x)        # (vocab_size, T, B)

        @return logits
    end
end