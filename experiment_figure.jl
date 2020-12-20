using CSV, DataFrames, GLM, Statistics, Plots
include("data_utils.jl")

datapath = "data/raw/baseline_b.csv"
coef_base, data = run_ols(datapath)
""

datapath="data/raw/e_step1_all.csv"
beta_1 = run_gradient(datapath, coef_base)

blist = ["data/raw/baseline_c.csv", "data/raw/baseline_d.csv"]
e1list = ["data/raw/e_step1_c.csv", "data/raw/e_step1_d.csv"]
e2list = ["data/raw/e_step2_all.csv"]

mse_base = o_mse_calc(blist, coef_base)
mse_1 = o_mse_calc(e1list, coef_base)
mse_2 = o_mse_calc(e2list, beta_1)
mse_check = o_mse_calc(blist, beta_1)

plot(["Risk Min., Baseline", "Naive Risk Min.", "Iterative Learning"], [ mse_base, mse_1, mse_2], seriestype=:bar, ylim =[500, 1150],label=nothing, ylabel="Out of Sample MSE")
savefig("figures/omse.pdf")
