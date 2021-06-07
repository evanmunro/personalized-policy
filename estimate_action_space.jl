includet("simulation_model.jl")
using StatsBase, GLM, DataFrames, LinearAlgebra, Plots


function estimateOptimalPolicy()
    players = SCTypeData(10000)
    d = 10000
    βs = rand(Uniform(0.5, 1.1), d)

    μs = zeros(d, 2)
    σs = zeros(d, 4)


    for i in 1:d
        data = ObservedData(players, [0, βs[i]])
        σs[i, :] = reshape(cov([data.y data.x]), (1, 4))
        μs[i, 1] = mean(data.y)
        μs[i, 2] = mean(data.x)
     end

    display(plot(βs, σs[:, 4], seriestype=:scatter))
    μ_coef = [coef(lm(@formula(μ ~ β + β^2), DataFrame(μ=μs[:, i], β=βs))) for i in 1:2]
    σ_coef = [coef(lm(@formula(σ ~ β + β^2), DataFrame(σ=σs[:, i], β=βs))) for i in 1:4]
    function loss(β)
        μb = [ dot(μ_coef[i], [1, β, β^2]) for i in 1:2]
        σb = reshape([ dot(σ_coef[i],[1, β, β^2]) for i in 1:4], (2,2))
        samples = rand(MvNormal(μb, σb), 1000000)
        β0 = mean(samples[1, :] .- samples[2, :].*β)
        return mean((samples[1, :] .- β0 .- samples[2, :]*β).^2)
    end

    βstar = Optim.optimize(loss, 0.0, 1.0).minimizer
    return βstar
end

#then we estimate a linear function for mean and variance of the x variables

# then we optimize the loss function when we integrate over a normal distribution
