using CategoricalArrays, DataFrames
include("model.jl")

function summarize_data(datapath)
    data = DataFrame!(CSV.File(datapath))
    recode_raw_survey_data!(data)
    println(describe(data, :mean, :std; cols=["Q1", "Q2", "Q3", "Q4"]))
    println(length(data.Q1))
end

function recode_raw_survey_data!(df)
    recode!(df.Q1, 1=>5.0, 2=> 15.0, 3=> 25.0, 4=> 35.0, 5=> 45.0, 6=>55.0,
                    7=> 65.0, 8=>75.0, 9=>85.0, 10=> 95.0, 11=> 125.0, 12=> 150.0)
    recode!(df.Q2, 1=> 21.0, 2=> 29.0, 3=> 39.0, 4=> 49.0, 5=> 59.0, 6=> 69.0, 7=> 79.0, 8=> 85.0)
    recode!(df.Q3, 1=> 10.0, 2=> 12.0, 3=> 13.0, 4=> 14.0, 5=> 16.0, 6=> 18.0, 7=> 20.0)
    recode!(df.Q4, 1=> 1.0, 2=>0.0)
end

function recode_perturb!(df, perturb)
    recode!(df.perturb1, 2=> 1.0, 1=> -1.0)
    df.perturb1 = df.perturb1.*perturb[1]
    recode!(df.perturb2, 2=> 1.0, 1=> -1.0)
    df.perturb2 = df.perturb2.*perturb[2]
    recode!(df.perturb3, 2=> 1.0, 1=> -1.0)
    df.perturb3 = df.perturb3.*perturb[3]
end

function check_bonus_payments(df, beta)
    dscale=1
    perturb = [0, beta[2:4]...]./3
    xmeans = [1, mean(df.Q2), mean(df.Q3), mean(df.Q4)]
    println("perturbs: ", perturb)
    println("mean prediction: ", sum(beta.*xmeans)*dscale)
    println("mean prediction at max perturb: ", sum((beta.+perturb).*xmeans)*dscale)
    println("mean prediction at min perturb: ", sum((beta.-perturb).*xmeans)*dscale)
    println("maximum possible prediction: ", sum((beta.+ perturb).*[1, 85, 20, 1])*dscale)
    println("minimum possible prediction: ", sum((beta.-perturb).*[1, 21, 10, 0])*dscale)
    println("minimum no perturb: ", sum((beta).*[1, 21, 10, 0])*dscale)
    println("maximum no perturb: ", sum(beta.*[1, 85, 20, 1])*dscale)
end

function run_ols(datapath)
    data = DataFrame!(CSV.File(datapath))
    recode_raw_survey_data!(data)
    ols = lm(@formula(Q1 ~ Q2 +Q3 +Q4), data)
    println(ols)
    return coef(ols), data
end

function run_gradient(datapath, β, α = [0.5, 1/80000.0, 1/150.0, 1/3.0])
    perturb = β[2:length(β)]./3
    data = DataFrame!(CSV.File(datapath))
    recode_raw_survey_data!(data)
    recode_perturb!(data, perturb)
    data_in = ExperimentData(data.Q1, Matrix(data[!, ["Q2", "Q3", "Q4"]]), Matrix(data[!, ["perturb1", "perturb2", "perturb3"]]))
    β_new = update_β_robust(data_in, β, α)
    return β_new
end


function o_mse_calc(files, beta)
    mses = zeros(length(files))
    n  = zeros(length(files))
    for (i, f) in zip(1:length(files), files)
        data = DataFrame!(CSV.File(f))
        recode_raw_survey_data!(data)
        x = Matrix([ones(size(data)[1]) data[!, ["Q2", "Q3", "Q4"]]])
        y = data.Q1
        yhat = x*beta
        mses[i] = mean((yhat .- y).^2)
        n[i] = length(yhat)
    end
    println(mses)
    return sum(mses.*n)/sum(n)
end
