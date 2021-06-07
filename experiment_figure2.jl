include("data_round2.jl")
using Plots

summary_stats("data/full_noincentive.csv")
summary_stats("data/full_bonus50.csv")
summary_stats("data/full_bonus100.csv")

g1n = calculate_gradients("data/step_from_naive.csv", [2.87701, 0.68888], 0.20)
g0n = get_b0_gradient("data/step_from_naive.csv", [2.87701, 0.68888], 0.20)
g1 = calculate_gradients("data/step_from_rrm.csv", [2.83594, 0.547041], 0.10)
g0 = get_b0_gradient("data/step_from_rrm.csv", [2.83594, 0.547041], 0.10)

r2_naive_insample = o_mse_calc(["data/full_noincentive.csv"], [2.87701, 0.68888])
noincentive = ["data/full_noincentive_out.csv"]
r2_naive = o_mse_calc(noincentive, [2.87701, 0.68888])
incentive = ["data/full_bonus100.csv"]
r2_naive_incentive = o_mse_calc(incentive, [2.87701, 0.68888])



b1 = 0.60
b0 = 2.89

r2_ic_ic = o_mse_calc(["data/step_out.csv"], [b0, b1])

plot(["Naive RM (IS)", "Naive RM, No Bonus", "Naive RM, Bonus", "OT-DE, Bonus"],
      [ r2_naive_insample, r2_naive, r2_naive_incentive, r2_ic_ic], seriestype=:bar,
        label=nothing, ylabel="R-Squared")
