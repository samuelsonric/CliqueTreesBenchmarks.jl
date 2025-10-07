include("../../TensorBenchmarks.jl")

using CliqueTrees
using OMEinsumContractionOrders
using PythonCall

using OMEinsumContractionOrders: EinCode

import JSON
import KaHyPar
import Metis

using .TensorBenchmarks

const DIR = joinpath("..", "..", "instances")

const FILE = first(ARGS)

const CTG = pyimport("cotengra")

const LABELS = (
    "CoTenGra",
    "OMEinsum (KaHyPar)",
    "OMEinsum (METIS)",
)

const PAIRS = (
    (Tuple{Py, Py, Py},    () -> CTG.HyperOptimizer(; reconf_opts=pydict(), max_time="rate:1e9", on_trial_error="ignore")),
    (Tuple{EinCode, Dict}, () -> HyperND(; dis = KaHyParND(), width = 50, imbalances = 100:10:800)),
    (Tuple{EinCode, Dict}, () -> HyperND(; dis = METISND(),   width = 50, imbalances = 100:10:800)),
)

function run()
    printrow(
        "circuit",
        "algorithm",
        "time complexity",
        "space complexity",
        "running time",
    )

    if endswith(FILE, ".json") && FILE in readdir(DIR)
        query, matrix, weights = TensorBenchmarks.read(joinpath(abspath(DIR), FILE))

        if all(weights .>= 2)
            dict = Dict{String, Dict{String, Float64}}()

            for (label, (T, f)) in zip(LABELS, PAIRS)            
                network = make(T, query, matrix, weights)
                tc = sc = time = typemax(Float64)

                try
                    optimizer = f()
                    result = solve(network, optimizer)
                    tc = timecomplexity(result)
                    sc = spacecomplexity(result)
                
                    optimizer = f()
                    time = @elapsed solve(network, optimizer)
                catch e
		    println(e)
                end

                dict[label] = Dict(
                    "tc" => tc,
                    "sc" => sc,
                    "time" => time,
                )

		printrow(
		    FILE,
		    label,
		    tc,
		    sc,
		    time,
		)
            end

            write(joinpath("results", FILE), JSON.json(dict))
        end
    end

    return
end

run()
