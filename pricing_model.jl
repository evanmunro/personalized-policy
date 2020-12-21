using Random, Distributions, Optim, Plots
include("simulation_model.jl")

struct PxTypeData <: TypeData
    v::Array{Float64}
    γ::Array{Float64}
    z::Array{Float64}
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
    β_fk = fk_solution(100000, PxTypeData, negRevenue)
    β_naive = naive_solution(100000, PxTypeData, fixedXObjective)
    tuner = ExperimentTuner(T, n, 0.2, [8e-2, 5e-4], [5.0, 0.4])
    methods = [IterativeUpdater(β_fk, tuner),
               IterativeUpdater(robustUpdate, tuner),
               IterativeUpdater(naiveUpdate, tuner),
               IterativeUpdater(β_naive, tuner)]
    runExperiment(tuner, methods, PxTypeData, revenue)
    βs = []
    fk_profits = mean(methods[1].π)
    plotmethods = [methods[1], methods[2], methods[4]]
    for m in methods
        if m in plotmethods
            push!(βs, mean(m.β[:, 2], dims=2))
        end
        println(mean(m.π))
        println(mean(m.π) - fk_profits)
        #push!(βs, m.β[:, 2])
        println(m.β[T, :])
    end
    plot( βs, xlabel = "t", ylabel="Price Discrimination",
    label=["Full Information" "Learning via Experiment"  "Naive Risk Min"], ylim = [0.0, 0.8])
    savefig("figures/price_sim.pdf")
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
