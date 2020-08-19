using CategoricalArrays, DataFrames

function recode_raw_survey_data!(df)
    recode!(df.Q1, 1=>5.0, 2=> 15.0, 3=> 25.0, 4=> 35.0, 5=> 45.0, 6=>55.0,
                    7=> 65.0, 8=>75.0, 9=>85.0, 10=> 95.0, 11=> 125.0, 12=> 150.0)
    recode!(df.Q2, 1=> 21.0, 2=> 29.0, 3=> 39.0, 4=> 49.0, 5=> 59.0, 6=> 69.0, 7=> 79.0, 8=> 85.0)
    recode!(df.Q3, 1=> 10.0, 2=> 12.0, 3=> 13.0, 4=> 14.0, 5=> 16.0, 6=> 18.0, 7=> 20.0)
    recode!(df.Q4, 1=> 1.0, 2=>0.0)
end

function check_bonus_payments(df, beta)
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
