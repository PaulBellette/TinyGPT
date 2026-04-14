# TinyGPT

This repo is a personal experiment in learning about transformers (late to the party but whatever). I wanted to understand the mechanics in a little more detail so I built this. 

In syntax.jl there is a really simple language to learn that makes expressions like

```julia
julia> using Random
julia> TinyGPT.decode_tokens(TinyGPT.generate_example(MersenneTwister(0)))
"C : 4 ; F : 3 ; D : 0 ; ? F = 3"
```

So the aim of the model is to learn to identify the key after the question mark and retrieve its value from the preceding context.

To try it with a little 2 layer model do something like

```julia
cfg = TinyGPTConfig(
    length(VOCAB),  # vocab size
    32,             # max seq len
    64,             # d_model
    4,              # n_heads
    256,            # d_ff
    2,              # n_layers
    0.0f0,          # dropout probability
)
```

```julia
model = TinyGPTModel(cfg)
```

```julia
ps, st, losses, accs = train!(model; steps=2000, batch_size=64, lr=1f-3)
```

Initial prototyping and some implementation iterations were developed collaboratively with ChatGPT (OpenAI), then adapted and organized into this repository by Paul Bellette.