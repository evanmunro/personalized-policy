using Random, Combinatorics, Distributions, StatsBase

struct Covariates
    d1::Array{Float64}
    d0::Array{Float64}
    cost::Array{Float64}
    d::Int
end

function Covariates(d::Int64)::Covariates
    dcost = Normal(d)
    cost = abs.(rand(dcost,d))
    db = Beta(1)
    dHigh = zeros(d)
    dLow = zeros(d)
    for i in 1:d
        ds = rand(db,2)
        dHigh[i] = maximum(ds)
        dLow[i] = minimum(ds)
    end
    return Covariates(dHigh,dLow,cost,d)
end

function manipulation_benefit()
    return 0
end

function sampleX(d1::Array{Float64})
    x = zeros(length(d1))
    for i in 1:length(d1)
        x[i]= sample([0,1],ProbabilityWeights([1-d1[i],d1[i]]))
    end
    return x
end

function sampleX(distNull::Covariates, model)
    dNoManip = distNull.d0
    
    return sampleX(distNull.d0)
end

function generate_sample(N::Int64,distNull::Covariates,model=Nothing,fracH=0.3)
    outcomeProb = ProbabilityWeights([1-fracH, fracH])
    d = distNull.d
    y = zeros(N)
    x = zeros(N,d)
    for i in 1:N
        y[i] = sample([0,1],outcomeProb)
        if y[i] == 1
            # high types don't manipulate, they choose X normally
            x[i,:] = sampleX(distNull.d1)
        else
            # low types might manipulate, take that into account
            if model == Nothing
                x[i,:] = sampleX(distNull.d0)
            else
                x[i,:] = sampleX(distNull,model)
            end
        end
    end
    return x,y
end
