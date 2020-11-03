using Random, Distributions, Optim, Plots, RollingFunctions

function broadcastArray(β::Array{Float64}, n)
    if ndims(β)== 1
        β  = zeros(n, 2) .+ transpose(β)
    end
    return β
end
struct PxTypeData
    v::Array{Float64}
    γ::Array{Float64}
    z::Array{Float64}
end

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
    function fixedβ(a, types::PxTypeData, t, tuner::ExperimentTuner)
        return βfixed
    end
    IterativeUpdater(fixedβ, β, π)
end

function ExperimentTuner(T, n)
    return ExperimentTuner(T, n, 0.2, [8e-2, 5e-4], [5.0, 0.25])
end


function PxTypeData(n=1000)
    z = rand(Uniform(10, 20), n)
    v = z .+ rand(Normal(0, 2), n) .+5
    γ = rand(Uniform(0, 3), n)
    return PxTypeData(v, γ, z)
end

function ObservedData(types::PxTypeData, β::Array{Float64})
    β = broadcastArray(β, length(types.γ))
    x = (types.z  .- types.γ.*β[:, 2].*(types.v .- β[:, 1]))./(1 .- β[:, 2].^2 .*types.γ)
    w = max.(β[:, 1] .+ β[:, 2].*x, 0)
    y = max.(types.v .- w, 0)
    return ObservedData(x, w, y)
end

function negRevenue(β::Array{Float64, 1}, types::PxTypeData)
    data = ObservedData(types, β)
    Π = sum(data.w .* data.y)
    return -Π
end

function revenue(β::Array{Float64}, types::PxTypeData)
    data = ObservedData(types, β)
    π = data.w .* data.y
    return π
end

function robustUpdate(β, types::PxTypeData, t, tuner::ExperimentTuner)
    step = tuner.α/(2+log(t))
    ϵ = tuner.ξ.*rand([-1,1], (tuner.n, length(β)))
    βp = β' .+ ϵ
    data = ObservedData(types, βp)
    π = revenue(βp, types)
    γ̂ = linReg(π, ϵ)[2:(length(β)+1)]
    return β .+ step.*γ̂
end

function fixedXObjective(β, x::Vector, types::PxTypeData)
    w = max.(β[1] .+ β[2] .* x, 0)
    y = max.(types.v .- w, 0)
    π = y .* w
    return -sum(π)
end

function naiveUpdate(β::Array{Float64}, types::PxTypeData, t::Int, tuner::ExperimentTuner)
    data = ObservedData(types, β)
    res = Optim.optimize(beta -> fixedXObjective(beta, data.x, types), β)
    return Optim.minimizer(res)
end

function runPxExperiment(T)
    n = 5000
    β_fk = fk_solution(100000)
    β_naive = naive_solution(100000)
    tuner = ExperimentTuner(T, n)
    methods = [IterativeUpdater(β_fk, tuner),
               IterativeUpdater(robustUpdate, tuner),
               IterativeUpdater(naiveUpdate, tuner),
               IterativeUpdater(β_naive, tuner)]
    runExperiment(tuner, methods)
    βs = []
    fk_profits = mean(methods[1].π)
    println(methods[4].π[20:30])
    println(methods[3].π[20:30])
    for m in methods
        push!(βs, rollmean(m.β[:, 2], 2))
        println(mean(m.π))
        println(mean(m.π) - fk_profits)
        #push!(βs, m.β[:, 2])
        println(m.β[T, :])
    end
    plot( βs, xlabel = "t", ylabel="Price Discrimination",
    label=["Full Knowledge" "Learning via Experiment" "Repeated Risk Min" "Naive Risk Min"], ylim = [0.0, 0.8])
end

function runExperiment(tuner::ExperimentTuner, methods::Array{IterativeUpdater})
    types = PxTypeData(tuner.n)
    for m in methods
        m.β[1, :] = m.update(tuner.β₀, types, 1, tuner)
        m.π[1] = mean(revenue(m.β[1, :], types))
    end
    for t in 2:tuner.T
        types = PxTypeData(tuner.n)
        for m in methods
            m.β[t, :] = m.update(m.β[t-1, :], types, t, tuner)
            m.π[t] = mean(revenue(m.β[t, :], types))
        end
    end
end

function naive_solution(n)
    types = PxTypeData(n)
    res = Optim.optimize(beta -> fixedXObjective(beta, types.z, types), [1, 0.0])
    return Optim.minimizer(res)
end

function fk_solution(n)
    types = PxTypeData(n)
    res = Optim.optimize(beta -> negRevenue(beta, types), [1, 0.0])
    bstar = Optim.minimizer(res)
    return bstar
end

#utilities and tests
function linReg(y, x)
    x = [ones(length(y)) x ]
    β = inv(x'*x)*x'*y
    return β
end

function test_analytic_x()
    v = 12.0; z = 10.0; γ = 1; β = [8, 0.1]
    xAnalytic = (z  .- γ.*β[2].*(v .- β[1]))./(1 .- β[2].^2 .*γ)
    res = Optim.optimize( xstar -> utility(xstar, v, γ, z, β),
                                                        [z], LBFGS())
    xNumerical = Optim.minimizer(res)
    (xNumerical[1] .- xAnalytic) <= 1e-5
end

function utility(x, v, γ, z, β)
    x = x[1]
    w = β[1] + β[2]*x
    y = v - w
    utility = (v-w)*y - 1/2*y^2 - 1/2*(x - z)^2/γ
    return -utility
end
