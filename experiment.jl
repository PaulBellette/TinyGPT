using Pkg
Pkg.activate(".")
using TinyGPT
using Random
using Statistics
#=

We have my two silly notions to test

1) weight sharing in the middle layer of the GPT may increase parameter count and accuracy
2) a symmteric decode block may increase stability

symmetric decoding is motivated by the analogy of Attention -> MLP looks like Lie-Trotter style splitting
so my test of this analogy is to push it towards a more stable symmetric update like Strang, 
    i.e. 1/2 Attention ->  MLP -> 1/2 Attention
If the operator view is correct this should have better errror characteristics than Lie-Trotter

I'm going to test this on my synthetic grammar by watching time and iterations to learn to > 0.9 accuracy 
with models with the same parameter count
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

cfg_looped = TinyGPT.TinyRecycledGPTConfig(
    length(VOCAB),  # vocab size
    32,             # max seq len
    64,             # d_model
    4,              # n_heads
    256,            # d_ff
    1,              # n_in_layers
    3,              # n_middle_loops
    1,              # n_middle_layers
    1,              # n_out_layers
    0.0f0,          # dropout probability
)

#model 1 - Baseline
model_1 = TinyGPTModel(cfg_plain)

#model 2 - Ordinary Structure + Strang Decode
model_2 = TinyGPT.TinySymmetricGPTModel(cfg_plain)

#model 3 - Looped Structure + Normal Decode
model_3 = TinyGPT.TinyRecycledGPTModel(cfg_looped)

#model 4 - Looped Structure + Strang Decode
model_4 = TinyGPT.TinyRecycledSymmetricGPTModel(cfg_looped)

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