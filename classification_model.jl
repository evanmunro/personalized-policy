using Random, Distributions
include("simulation_utils.jl")

struct SCTypeData <: TypeData
    γ::Array{Float64}
    z::Array{Float64}
    r::Array{Float64}
end

function SCTypeData(n=1000)
    z = rand(Normal(), n)
    r = rand(Normal(), n)
    γ = rand(Uniform(0, 8), n)
    return SCTypeData(γ, z, r)
end

function ObservedData(types::SCTypeData, β::Array{Float64})
    β = broadcastArray(β, length(types.γ))
    x = types.z .+ types.γ.*(β[:, 2])
    y = types.z .+ types.r
    w = β[:, 1] .+ x.*β[:, 2]
    return ObservedData(x, w, y)
end

function negloss(β::Array{Float64, 1}, types::SCTypeData)
    data = ObservedData(types, β)
    Π = sum((data.y - data.w).^2)
    return Π
end

function loss(β::Array{Float64}, types::SCTypeData)
    data = ObservedData(types, β)
    π = (data.y - data.w).^2
    return -π
end

function robustUpdate(β, types::SCTypeData, t, tuner::ExperimentTuner)
    step = tuner.α/(2+log(t))
    ϵ = tuner.ξ.*rand([-1,1], (tuner.n, length(β)))
    βp = β' .+ ϵ
    data = ObservedData(types, βp)
    π = loss(βp, types)
    γ̂ = linReg(π, ϵ)[2:(length(β)+1)]
    return β .+ step.*γ̂
end

function fixedXObjective(β, x::Vector, types::SCTypeData)
    w = β[1] .+ β[2] .* x
    y = types.z .+ types.r
    π = (y .- w).^2
    return sum(π)
end

function naiveUpdate(β::Array{Float64}, types::SCTypeData, t::Int, tuner::ExperimentTuner)
    data = ObservedData(types, β)
    res = Optim.optimize(beta -> fixedXObjective(beta, data.x, types), β)
    return Optim.minimizer(res)
end

function runSCExperiment(T)
    n = 5000
    β_fk = fk_solution(100000, SCTypeData, negloss)
    β_naive = naive_solution(100000, SCTypeData, fixedXObjective)
    tuner = ExperimentTuner(T, n, 0.2, [8e-2, 5e-4], β_naive)
    methods = [IterativeUpdater(β_fk, tuner),
               IterativeUpdater(robustUpdate, tuner),
               IterativeUpdater(naiveUpdate, tuner),
               IterativeUpdater(β_naive, tuner)]
    runExperiment(tuner, methods, SCTypeData, loss)
    βs = []
    fk_profits = mean(methods[1].π)
    for m in methods
        push!(βs, rollmean(m.β[:, 2], 1))
        #print regret
        println(mean(m.π))
        println(mean(m.π) - fk_profits)
        println(m.β[T, :])
    end
    plot( βs, xlabel = "t", ylabel="Prediction Weight",
    label=["Full Information" "Learning via Experiment" "Repeated Risk Min" "Naive Risk Min"], ylim = [0.0, 0.8])
end
