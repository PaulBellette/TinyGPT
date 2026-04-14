module TinyGPT

export TinyGPTConfig, TinyGPTModel, VOCAB, train!

using Lux
using LuxLib
using Random
using Optimisers
using Zygote
using Statistics
using NNlib

include("syntax.jl")
include("models.jl")
include("training.jl")

end
