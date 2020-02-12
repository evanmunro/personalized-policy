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

function sampleX(distNull::Covariates, model,benefit=10)
    d1 = distNull.d0
    manip = 0
    # true x's
    x = zeros(length(d1))
    for i in 1:length(d1)
        x[i]= sample([0,1],ProbabilityWeights([1-d1[i],d1[i]]))
    end

    # manipulated
    for i in 1:length(d1)
        deriv = coef(model)[i+1]
        if x[i] == 0
            if benefit*deriv > distNull.cost[i]
                x[i] = 1
                manip = 1
            end
        end
    end
    return x,manip
end

function generate_sample(N::Int64,distNull::Covariates,mProb=0,model=Nothing,fracH=0.3)
    outcomeProb = ProbabilityWeights([1-fracH, fracH])
    d = distNull.d
    y = zeros(N)
    x = zeros(N,d)
    manipulators = 0
    for i in 1:N
        y[i] = sample([0,1],outcomeProb)
        if y[i] == 1
            # high types don't manipulate, they choose X normally
            x[i,:] = sampleX(distNull.d1)
        else
            # low types might manipulate, take that into account
            knows = sample([0,1],ProbabilityWeights([1-mProb,mProb]))
            if (model == Nothing) || knows== 0
                x[i,:] = sampleX(distNull.d0)
            else
                x[i,:],manip = sampleX(distNull,model)
                manipulators += manip
            end
        end
    end
    return x,y, manipulators
end
