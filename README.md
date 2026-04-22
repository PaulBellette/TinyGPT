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
ps, st, losses, accs, batch_timings = train!(model; steps=2000, batch_size=64, lr=1f-3)
```
## Small Architecture Experiment

As a small side experiment, I tested two slightly silly architectural ideas on the synthetic retrieval grammar in this repo.

The first was to recycle a shared inner block instead of using only distinct layers, in the hope that repeated application of the same transform might improve parameter efficiency.

The second was to use a symmetric decoding block motivated by an operator-splitting analogy. A standard transformer block looks a bit like

`Attention -> MLP`

which, if you cross your eyes a little, looks like a Lie-Trotter scheme. So the extension here was to push that toward a Strang-like split,

`1/2 Attention -> MLP -> 1/2 Attention`

with the loose intuition that this may produce a more stable update than the asymmetric form. This is not meant as a deep claim, just a small empirical poke at an idea that popped up when considering the mechanics of transformers.

## Conditions

I compared four small models with the same parameter count:

- baseline TinyGPT
- baseline TinyGPT with symmetric decode block
- recycled inner-loop TinyGPT
- recycled inner-loop TinyGPT with symmetric decode block

For this shakedown run I used 20 random seeds, a maximum of 50,000 training iterations, and early stopping once validation accuracy exceeded 0.9 on the toy grammar task. I recorded the first threshold crossing and the mean batch time for each run. In the following table, success within budget means that greater than 0.9 accuracy was found within 50k iterations.

## Summary

| Model | Median threshold index | Successes within budget |
|---|---:|---:|
| Baseline | 203.0 | 15 / 20 |
| Symmetric decode | 148.0 | 20 / 20 |
| Recycled inner loop | 141.5 | 12 / 20 |
| Recycled + symmetric decode | 130.0 | 19 / 20 |

A rough read of this is:

- the baseline was clearly the weakest condition
- the symmetric decode variant was the most reliable
- the recycled-only model could learn quickly when it worked, but was much less reliable across seeds
- the recycled + symmetric model had the best median iterations-to-threshold, but also the highest per-step runtime cost

So the strongest signal from this little experiment was not really “weight sharing wins”, but rather that symmetric decoding appeared to improve optimization stability on this toy task.

## Caveats

This is a toy-grammar shakedown, not a serious benchmark, and seed-to-seed variance was large. I would not claim any general transformer result from this.

Still, I think it is worth keeping in the repo because it shows the sort of lightweight empirical loop I wanted from this project: have an intuition, make a small ablation, measure something concrete, and leave the result in the open even if it is messy.

# Acknowledgement

Initial prototyping and some implementation iterations were developed collaboratively with ChatGPT (OpenAI), then adapted and organized into this repository by Paul Bellette.
