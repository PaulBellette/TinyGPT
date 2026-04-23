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
## Small Architecture Experiment 1 - Symmetry and Iterated Innner Layers

As a small side experiment, I tested two slightly silly architectural ideas on the synthetic retrieval grammar in this repo.

The first was to recycle a shared inner block instead of using only distinct layers, in the hope that repeated application of the same transform might improve parameter efficiency.

The second was to use a symmetric decoding block motivated by an operator-splitting analogy. A standard transformer block looks a bit like

`Attention -> MLP`

which, if you cross your eyes a little, looks like a Lie-Trotter scheme. So the extension here was to push that toward a Strang-like split,

`1/2 Attention -> MLP -> 1/2 Attention`

with the loose intuition that this may produce a more stable update than the asymmetric form. This is not meant as a deep claim, just a small empirical poke at an idea that popped up when considering the mechanics of transformers.

### Conditions

I compared four small models with the same parameter count:

- baseline TinyGPT
- baseline TinyGPT with symmetric decode block
- recycled inner-loop TinyGPT
- recycled inner-loop TinyGPT with symmetric decode block

For this shakedown run I used 20 random seeds, a maximum of 50,000 training iterations, and early stopping once validation accuracy exceeded 0.9 on the toy grammar task. I recorded the first threshold crossing and the mean batch time for each run. In the following table, success within budget means that greater than 0.9 accuracy was found within 50k iterations.

### Summary

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

## Small Architecture Experiment 2 - Scratch Pad Latent

As a second small architectural poke, I compared two ways of adding hidden scratch-space to the model against the baseline TinyGPT.

The first variant, `PrefixMemory`, was motivated by the thought that a standard transformer residual stream may be doing too many jobs at once: representing the current token, carrying contextual information, and also acting as a kind of temporary workspace. In this version, each position writes a candidate memory vector from its current hidden state, and these writes are then combined into a causal weighted prefix average. That gives each token position access to a learned running summary of the prefix up to that point, which is then projected back into the token stream before the MLP update. So the model gets an additional hidden memory channel without breaking autoregressive causality.

The second variant, `PrependedLatent`, was motivated by a slightly different idea: instead of injecting a running memory summary, prepend a small bank of learned latent tokens to the visible sequence and let the model attend to them directly. These latent slots live in the same `d_model` space as the normal token representations, but are not decoded as output tokens. The hope here was that they might act like hidden registers or scratch space that later visible tokens could read from through the ordinary attention mechanism. In the implementation here, the latent tokens are trainable and are threaded through the stack alongside the visible sequence, so this version is best thought of as adding hidden token-like workspace rather than a separate causal summary channel.

### Conditions

I compared three models:

- baseline TinyGPT
- `PrefixMemory`
- `PrependedLatent`

As above, I used 20 random seeds, a maximum training budget of 50,000 iterations, and early stopping once validation accuracy exceeded 0.9. I recorded the first threshold crossing and the mean batch time for each run.

### Summary

| Model | Median threshold index | Successes within budget | Mean batch time (s) |
|---|---:|---:|---:|
| Baseline | 205.5 | 14 / 20 | 1.178 |
| PrefixMemory | 176.0 | 15 / 20 | 1.475 |
| PrependedLatent | 167.0 | 11 / 20 | 1.218 |

A rough read of this is:

- both latent variants could reach threshold faster than the baseline when they worked
- `PrefixMemory` was the most reliable of the three, but also had the highest per-step runtime cost
- `PrependedLatent` had the best median iterations-to-threshold, but was less reliable across seeds
- at least on this toy task, the added scratch-space seems to change optimization behaviour more than it improves robustness

## Small Architecture Experiment 3 - Low Rank MLP

A third experiment was to replace the standard feedforward block with a simple low-rank factorization, motivated by the thought that the post-attention transform on this toy task might live in a smaller subspace than a full dense MLP suggests. I was also interested in this because the MLPs make up a large fraction of the parameter count in large models, so there may in principle be a straightforward opportunity to reduce that cost through explicit factorization.

### Conditions

I compared four models:

- baseline TinyGPT
- low-rank MLP with rank 32
- low-rank MLP with rank 16
- low-rank MLP with rank 8

Again, I used 20 random seeds, a maximum of 50,000 training iterations, and early stopping once validation accuracy exceeded 0.9. I recorded the first threshold crossing and the mean batch time for each run.

### Summary

| Model | Median threshold index | Successes within budget | Mean batch time (s) |
|---|---:|---:|---:|
| Baseline | 140.0 | 11 / 20 | 1.206 |
| Rank 32 | 224.0 | 3 / 20 | 1.353 |
| Rank 16 | 267.0 | 8 / 20 | 1.330 |
| Rank 8 | 199.0 | 11 / 20 | 1.303 |

A rough read of this is:

- none of the low-rank variants beat the baseline on this task
- the rank-32 condition was especially weak, both in speed and reliability
- rank-8 recovered baseline-level reliability, but still trained more slowly
- on this synthetic retrieval grammar, the standard dense MLP seems to be a better default than this very simple low-rank parameterization

So did this experiment move the state of trasformer technology forward? No. But did I learn anything? Also no.

## Caveats

This is a toy-grammar shakedown, not a serious benchmark, and seed-to-seed variance was large. Actually, the biggest confounder of any results is this seed to seed variance in performance. I am actually super intrigued by the loss behaviour of this model as it rapidly moves to 0.5 accuracy, settles in and then maybe eventually finds a hole in the loss landscape and jumps up to 1.0 accuracy. It motivates more experiments on weight initialisation and optimisation behavious. That said, I would not claim any general transformer results from these little experiments.

Still, I think they are worth keeping in the repo because it shows the sort of lightweight empirical loop I wanted from this project: have an intuition, make a small ablation, measure something concrete, and leave the result in the open even if it is messy. A research notebook in the wild.

# Acknowledgement

Initial prototyping and some implementation iterations were developed collaboratively with ChatGPT (OpenAI), then adapted and organized into this repository by Paul Bellette.
