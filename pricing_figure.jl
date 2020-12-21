include("pricing_model.jl")

Random.seed!(1)
runPxExperiment(500)
