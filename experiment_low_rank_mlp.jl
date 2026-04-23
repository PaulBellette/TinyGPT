using Pkg
Pkg.activate(".")
using TinyGPT
using Random
using Statistics
#=

We have yet another silly notion to test

Low rank formulation of MLP matrices at different max rank
=#

cfg_plain = TinyGPTConfig(
    length(VOCAB),  # vocab size
    32,             # max seq len
    64,             # d_model
    4,              # n_heads
    256,            # d_ff
    3,              # n_layers
    0.0f0,          # dropout
)

cfg_r32 = TinyGPT.TinyOuterProductMLPGPTConfig(
    length(VOCAB),  # vocab size
    32,             # max seq len
    64,             # d_model
    4,              # n_heads
    256,            # d_ff
    3,              # n_layers
    0.0f0,          # dropout
    32              # r
)

cfg_r16 = TinyGPT.TinyOuterProductMLPGPTConfig(
    length(VOCAB),  # vocab size
    32,             # max seq len
    64,             # d_model
    4,              # n_heads
    256,            # d_ff
    3,              # n_layers
    0.0f0,          # dropout
    16              # r
)

cfg_r8 = TinyGPT.TinyOuterProductMLPGPTConfig(
    length(VOCAB),  # vocab size
    32,             # max seq len
    64,             # d_model
    4,              # n_heads
    256,            # d_ff
    3,              # n_layers
    0.0f0,          # dropout
    8               # r
)


#model 1 - Baseline
model_1 = TinyGPTModel(cfg_plain)

#model 2 - Ordinary Structure + low rank mlp 32
model_2 = TinyGPT.TinyOuterProductMLPGPTModel(cfg_r32)

#model 3 - Ordinary Structure + low rank mlp 16
model_3 = TinyGPT.TinyOuterProductMLPGPTModel(cfg_r16)

#model 4 - Ordinary Structure + low rank mlp 8
model_4 = TinyGPT.TinyOuterProductMLPGPTModel(cfg_r8)

Random.seed!(0)
seeds = rand(Int,20)

model_1_threshold_iterations = []
model_1_mean_batch_time = []
model_1_loss_hist = []

model_2_threshold_iterations = []
model_2_mean_batch_time = []
model_2_loss_hist = []

model_3_threshold_iterations = []
model_3_mean_batch_time = []
model_3_loss_hist = []

model_4_threshold_iterations = []
model_4_mean_batch_time = []
model_4_loss_hist = []

max_iterations = 50000

for (index, seed) in enumerate(seeds)
    @info "seed number $(index)"
    @info "model 1 training"
    _, _, losses, accs, batch_timings = train!(model_1; steps=max_iterations, batch_size=64, lr=1f-3, seed = seed, backend=  "gpu", early_stop = true)
    push!(model_1_threshold_iterations, findfirst(accs .> 0.9))
    push!(model_1_mean_batch_time, mean(batch_timings[2:end]))
    push!(model_1_loss_hist, losses)
    
    @info "model 2 training"
    _, _, losses, accs, batch_timings = train!(model_2; steps=max_iterations, batch_size=64, lr=1f-3, seed = seed, backend=  "gpu", early_stop = true)
    push!(model_2_threshold_iterations, findfirst(accs .> 0.9))
    push!(model_2_mean_batch_time, mean(batch_timings[2:end]))
    push!(model_2_loss_hist, losses)

    @info "model 3 training"
    _, _, losses, accs, batch_timings = train!(model_3; steps=max_iterations, batch_size=64, lr=1f-3, seed = seed, backend=  "gpu", early_stop = true)
    push!(model_3_threshold_iterations, findfirst(accs .> 0.9))
    push!(model_3_mean_batch_time, mean(batch_timings[2:end]))
    push!(model_3_loss_hist, losses)

    @info "model 4 training"
    _, _, losses, accs, batch_timings = train!(model_4; steps=max_iterations, batch_size=64, lr=1f-3, seed = seed, backend=  "gpu", early_stop = true)
    push!(model_4_threshold_iterations, findfirst(accs .> 0.9))
    push!(model_4_mean_batch_time, mean(batch_timings[2:end]))
    push!(model_4_loss_hist, losses)
end