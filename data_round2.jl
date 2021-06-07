using CategoricalArrays, DataFrames, GLM, CSV, Statistics

function recode_raw_survey_data!(df)
    recode!(df.Q3, 1=> 10.0, 2=> 12.0, 3=> 13.0, 4=> 14.0, 5=> 16.0, 6=> 18.0, 7=> 20.0)
    recode!(df.perturb1, 2=> 1.0, 1=> -1.0)
end

function confusion(var1, var2)
    println("11: ", sum(var1.*var2)/length(var1))
    println("10: ", sum(var1.*(1 .-var2)/length(var1)))
    println("01: ", sum((1 .-var1).*var2/length(var1)))
    println("00: ", sum((1 .-var1).*(1 .-var2)/length(var1)))
end

function clean_data(datapath)
    data = DataFrame!(CSV.File(datapath))
    recode_raw_survey_data!(data)
    recode!(data.Q11, missing => -100)
    data.correct = (data.Q11 .≈ 3) .+ (data.Q11 .≈ -0.5)
    println(nrow(data))
    return head(data, 316)
end

function summary_stats(datapath)
    data = clean_data(datapath)
    for var in [data.Q10, data.Q12, data.Q3, data.correct]
        println(mean(var), " (", std(var), ") ")
    end
    print(lm(@formula(Q10 ~ correct ), data))
    #println("Education and Skill: ", cor(data.Q3, data.Q12))
    #println("Correct and Outcome: ", cor(data.correct, data.Q10))
end

function linReg(y, x)
    x = [ones(length(y)) x ]
    β = inv(x'*x)*x'*y
    return β
end


function calculate_gradients(datapath, β, psize)
    data = clean_data(datapath)
    yhat = β[1] .+ (data.perturb1.*psize .+ β[2]).*data.correct
    π = -1 .*(yhat .- data.Q10).^2
    gradient = linReg(π, data.perturb1 .* psize)[2]
    println("Gradient: ", gradient)
    return gradient
end

function get_b0_gradient(datapath, β, psize)
    data = clean_data(datapath)
    net = mean(data.Q10) - β[1] - mean(data.correct.*β[2])
    println(net)
    return net
end

function get_intercept(datapath, β)
    data = clean_data(datapath)
    β0  = mean(data.Q10) - mean(β.*data.correct)
    return β0
end

function o_mse_calc(files, beta)
    mses = zeros(length(files))
    n  = zeros(length(files))
    for (i, f) in zip(1:length(files), files)
        data = clean_data(f)
        x = Matrix([ones(size(data)[1]) data[!, ["correct"]]])
        y = data.Q10
        yhat = x*beta
        mses[i] =  1 - mean((yhat .- y).^2)/var(y)
        println("MSE: ", mean((yhat .- y).^2))
        n[i] = length(yhat)
    end
    println(mses)
    return sum(mses.*n)/sum(n)
end
