using Random, Distributions, Optim, Plots, Statistics

abstract type TypeData end

struct ObservedData
    x::Array{Float64}
    w::Array{Float64}
    y::Array{Float64}
end

struct ExperimentTuner
    T::Int
    n::Int
    ξ::Float64
    α::Array{Float64}
    β₀::Array{Float64}
end

mutable struct IterativeUpdater
    update::Function
    β::Array{Float64, 2}
    π::Array{Float64, 1}
end

function IterativeUpdater(update::Function, tuner::ExperimentTuner)
    k = length(tuner.β₀)
    β = zeros(Float64, (tuner.T, k))
    π = zeros(Float64, tuner.T)
    IterativeUpdater(update, β, π)
end

function IterativeUpdater(βfixed::Array{Float64}, tuner::ExperimentTuner)
    k = length(tuner.β₀)
    β = zeros(Float64, (tuner.T, k))
    π = zeros(Float64, tuner.T)
    function fixedβ(a, types::TypeData, t, tuner::ExperimentTuner)
        return βfixed
    end
    IterativeUpdater(fixedβ, β, π)
end

function broadcastArray(β::Array{Float64}, n)
    if ndims(β)== 1
        β  = zeros(n, 2) .+ transpose(β)
    end
    return β
end

function runExperiment(tuner::ExperimentTuner, methods::Array{IterativeUpdater},
                       typeDGP::Type, objective::Function)
    types = typeDGP(tuner.n)
    for m in methods
        m.β[1, :] = m.update(tuner.β₀, types, 1, tuner)
        m.π[1] = mean(objective(m.β[1, :], types))
    end
    for t in 2:tuner.T
        types = typeDGP(tuner.n)
        for m in methods
            m.β[t, :] = m.update(m.β[t-1, :], types, t, tuner)
            m.π[t] = mean(objective(m.β[t, :], types))
        end
    end
end

function naive_solution(n::Int, typeDGP::Type, naiveObjective::Function)
    types = typeDGP(n)
    res = Optim.optimize(beta -> naiveObjective(beta, types.z, types), [1, 0.0])
    return Optim.minimizer(res)
end

function fk_solution(n::Int, typeDGP::Type, negObjective::Function)
    types = typeDGP(n)
    res = Optim.optimize(beta -> negObjective(beta, types), [1, 0.0])
    bstar = Optim.minimizer(res)
    return bstar
end

#utilities and tests
function linReg(y, x)
    x = [ones(length(y)) x ]
    β = inv(x'*x)*x'*y
    return β
end


#specific types
struct SCTypeData <: TypeData
    γ::Array{Float64}
    z::Array{Float64}
    r::Array{Float64}
end

function SCTypeData(n=1000)
    z = rand(Normal(0, 1), n)
    r = rand(Normal(0, 1), n)
    γ = rand(Uniform(0, 1.5), n)
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

function SCUpdate(β, types::SCTypeData, t, tuner::ExperimentTuner)
    step = tuner.α[1]/2
    ϵ = [ zeros(tuner.n) tuner.ξ.*rand([-1,1], (tuner.n, length(β)-1))]
    βp = β' .+ ϵ
    data = ObservedData(types, βp)
    π = loss(βp, types)

    γ̂ = linReg(π, ϵ[:, 2])[2]
    β_new = zeros(2)
    β_new[2] = β[2] + step*γ̂
    β_new[1] = mean(data.y)  - mean(data.x.*β_new[2])
    return β_new
end
function robustUpdate(β, types::SCTypeData, t, tuner::ExperimentTuner)
    step = tuner.α/(2)
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
    return linReg(data.y, data.x)
end

function runSCExperiment(T)
    n = 5000
    β_fk = fk_solution(100000, SCTypeData, negloss)

    β_naive = naive_solution(100000, SCTypeData, fixedXObjective)
    tuner = ExperimentTuner(T, n, 0.075, [3e-3], [-0.5, 0.87])
    methods = [IterativeUpdater(β_fk, tuner),
               IterativeUpdater(SCUpdate, tuner),
               IterativeUpdater(naiveUpdate, tuner),
               IterativeUpdater(β_naive, tuner)]
    runExperiment(tuner, methods, SCTypeData, loss)
    βs = []
    fk_profits = mean(methods[1].π)

    for m in methods
        push!(βs, mean(m.β[:, 2], dims=2))
        #print regret
        println(mean(m.π))
        println(mean(m.π) - fk_profits)
        println(m.β[T, :])
    end
    plot( βs, xlabel = "t", ylabel="Prediction Weight",
    label=["Full Information" "Learning via Experiment" "Repeated Risk Min" "Naive Risk Min"], ylim = [0.7, 1.2])
    savefig("figures/sc_sim.pdf")
end
