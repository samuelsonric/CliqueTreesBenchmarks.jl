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

    for file in ARGS
        if endswith(file, ".json") && file in readdir(DIR)
            run(file)
        end
    end

    return
end

function run(file)
    query, matrix, weights = TensorBenchmarks.read(joinpath(abspath(DIR), file))

    if all(weights .>= 2)
        dict = Dict{String, Dict{String, Float64}}()

        for (label, (T, f)) in zip(LABELS, PAIRS)            
            tc, sc, time = runalg(T, f, query, matrix, weights)
            tc, sc, time = runalg(T, f, query, matrix, weights) 

            dict[label] = Dict(
                "tc" => tc,
                "sc" => sc,
                "time" => time,
            )

            printrow(
                file,
                label,
                tc,
                sc,
                time,
            )
        end

        write(joinpath("results", file), JSON.json(dict))
    end

    return
end

function runalg(T, f, query, matrix, weights)
    network = make(T, query, matrix, weights)
    optimizer = f()
    tc = sc = time = typemax(Float64)

    try
        time = @elapsed result = solve(network, optimizer)
        tc = timecomplexity(result)
        sc = spacecomplexity(result)
    catch e
        println(e)
    end

    return tc, sc, time
end

run()
