using Distributions, Random, JuMP, Ipopt
using LinearAlgebra
using Suppressor

struct ExperimentData
    y::Array{Float64}
    x
    ξ
end

function frankel_generator(β, ξ)
    n, d = size(ξ)
    β = β[2:(d+1)]
    γ = rand(Uniform(0, 8), n)
    m = 1
    z = rand(Normal(), (n, d))
    e = rand(Normal(), n)
    x = z .+ m.*γ.*(ξ.+β')
    y = z*ones(d) .+ e
    return ExperimentData(y, x, ξ)
end

function mock_survey_coder(β, ξ, recode, n)
    m = length(recode)
    γ = rand(Uniform(0, 8), n)
    z = rand(1:m, n)
    x = Int.(floor.(z .+ γ.*(ξ.+β)))
    x[x.>m] .= m
    x[x .< m] .=1
    x = recode[x]
    z = recode[z]
    return z, x
end

function survey_generator(β, ξ)
    n, d = size(ξ)
    β = β[2:(d+1)]
    z1, x1 = mock_survey_coder(β[1], ξ[:,1], [21.0, 29.0, 39.0, 49.0, 59.0, 69.0, 79.0, 85.0], n)
    z2, x2 = mock_survey_coder(β[2], ξ[:,2], [10, 12, 13, 14, 16, 18, 20], n)
    z3, x3 = mock_survey_coder(β[3], ξ[:,3], [0, 1], n)
    #y = floor.((10.0 .+ [z1 z2 z3]*[0.1, 0.12, 10.0])/10.0)*10
    y = 10.0 .+ [z1 z2 z3]*[0.1, 0.12, 10.0]
    return ExperimentData(y, [x1 x2 x3], ξ)
end

function update_β_ols(data::ExperimentData, β, α=0)
    β_n = linReg(data.y, data.x)
    return β_n
end

function mse(data, β)
    return mean((data.y .- [ones(length(data.y)) data.x]*β).^2)
end

#20000, 100, 4
function update_β_robust(data::ExperimentData, β, α=[0.5, 1/80000.0, 1/150.0, 1/3.0])
    n, d = size(data.ξ)
    yhat = β[1] .+ sum((data.ξ .+ β[2:(d+1)]').*data.x, dims=2)

    ehat = data.y .- yhat
    dβ = [-2/n*sum((data.y .- yhat)), linReg(ehat.^2, data.ξ)[2:(d+1)]...]
    println(dβ)
    β_n = β .- α.*dβ
    β_n[1] = mean(data.y)  - mean(sum(β_n[2:(d+1)]'.*data.x, dims=2))
    return β_n
end

function run_simulation(generator, updater, β0::Array{Float64, 1}, n::Int, steps::Int)
    d = length(β0) - 1
    β_history = zeros(Float64, (steps, d+1))
    mse_history = zeros(Float64, steps)
    β_history[1, :] = β0
    for s in 2:steps
        η = β_history[s-1, 2:(d+1)]
        ξ = rand([-1,1], (n, d)).*η'
        data = generator(β_history[s-1, :], ξ)
        mse_history[s-1] = mse(data, β_history[s-1, :])
        β_history[s, :] = updater(data, β_history[s-1, :])
    end

    mse_history[steps] = mse_history[steps-1]
    return β_history, mse_history
end




res = Optim.optimize(beta -> calc_loss(beta, model, y, z, gamma),
                                        zeros(Float32, k))
bstar = Optim.minimizer(res)


function pxDGP()

end
function fk_solution()
    γ = rand([0, 0.5, 8], n)
    z = rand(Normal(), n)
    e = rand(Normal(0, 0.5), n)
    y = z .+ e
    model =  Model(Ipopt.Optimizer)
    @variable(model, b1)
    @variable(model, b0)
    @NLobjective(model, Min, 1/n*sum((y[i] - b1*(z[i]+γ[i]*b1)-b0)^2 for i in 1:n))
    @suppress_out begin
        optimize!(model)
    end
    return [value(b0) value(b1)]
end

function linReg(y, x)
    x = [ones(length(y)) x ]
    β = inv(x'*x)*x'*y
    return β
end
