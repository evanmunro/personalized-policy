includet("pricing_model.jl")

βstar = fk_solution()
types = PxTypeData()
data  = ObservedData(types, βstar)
sum(data.y .==0)
