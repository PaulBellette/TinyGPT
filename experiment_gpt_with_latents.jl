using Pkg
Pkg.activate(".")
using TinyGPT
using Random
using Statistics
#=

We have another silly notion to test

can we give the model access to a "scratch pad" latent to help with learning this task

I've tried to do this via two seperate means, prefix memory and prepended latent
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

cfg_latent = TinyGPT.TinyLatentGPTConfig(
    length(VOCAB),  # vocab size
    32,             # max seq len
    64,             # d_model
    4,              # n_heads
    256,            # d_ff
    3,              # n_layers
    4,              # latent dim
    0.0f0,          # dropout
)

#model 1 - Baseline
model_1 = TinyGPTModel(cfg_plain)

#model 2 - Ordinary Structure + memory prefix
model_2 = TinyGPT.TinyPrefixMemoryGPTModel(cfg_latent)

#model 3 - Ordinary Structure + latent prepended
model_3 = TinyGPT.TinyPrependedLatentGPTModel(cfg_latent)

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
end