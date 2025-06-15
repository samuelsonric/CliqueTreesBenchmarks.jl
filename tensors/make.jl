include("TensorBenchmarks.jl")
include("YaoQASMReader.jl")

using CairoMakie
using CliqueTrees
using DataFrames
using PythonCall
using EinExprs

import KaHyPar
import Metis

using .TensorBenchmarks
using .YaoQASMReader

const CTG = pyimport("cotengra")

const LABELS = (
    "EinExprs (METIS)",
    "EinExprs (KaHyPar)",
    "CoTenGra",
)

const PAIRS = (
    (SizedEinExpr,  () -> HyPar(METISND();   imbalances = 100:10:800)),
    (SizedEinExpr,  () -> HyPar(KaHyParND(); imbalances = 100:10:800)),
    (Py,            () -> CTG.HyperOptimizer()),
)

function printrow(circuit, algorithm, tc, sc, time)
    print(" | ")
    print(rpad(circuit, 30))
    print(" | ")
    print(rpad(algorithm, 20))
    print(" | ")
    print(rpad(tc, 20))
    print(" | ")
    print(rpad(sc, 20))
    print(" | ")
    print(rpad(time, 20))
    print(" | ")
    println()
    return
end

function run()
    dataframe = DataFrame(
        tc=Float64[],    # time complexity
        sc=Float64[],    # space complexity
        time=Float64[],  # run time
        file = String[], # circuit file
        label=String[],  # label
    )

    printrow(
        "circuit",
        "algorithm",
        "time complexity",
        "space complexity",
        "running time",
    )

    for file in readdir("circuits")[1:3]
        if endswith(file, ".txt")
            path = joinpath(@__DIR__, "circuits", file)
            circuit = yaocircuit_from_qasm(path)
                    
            for (label, (T, f)) in zip(LABELS, PAIRS)
                network = make(T, circuit)
                
                optimizer = f()
                result = solve(network, optimizer)
                tc = timecomplexity(result)
                sc = spacecomplexity(result)
                
                optimizer = f()
                time = @elapsed solve(network, optimizer)

                printrow(
                    file,
                    label,
                    tc,
                    sc,
                    time,
                )
               
                push!(dataframe, (
                    tc,
                    sc,
                    time,
                    file,
                    label,
                ))
            end
        end
    end

    return dataframe
end

function plot(dataframe)
    figure = Figure(size = (450, 250))

    axis = Axis(figure[1, 1]; ylabel = "time complexity", xticksvisible = false, xticklabelsvisible = false)

    tc1 = dataframe.tc[dataframe.label .== LABELS[1]]
    tc2 = dataframe.tc[dataframe.label .== LABELS[2]]
    tc3 = dataframe.tc[dataframe.label .== LABELS[3]]

    perm = sortperm(tc1)

    scatter!(axis, tc1[perm], color = :red)
    scatter!(axis, tc2[perm], color = :green)
    scatter!(axis, tc3[perm], color = :blue)

    axis = Axis(figure[2, 1]; ylabel = "space complexity", xticksvisible = false, xticklabelsvisible = false)

    sc1 = dataframe.sc[dataframe.label .== LABELS[1]]
    sc2 = dataframe.sc[dataframe.label .== LABELS[2]]
    sc3 = dataframe.sc[dataframe.label .== LABELS[3]]

    perm = sortperm(sc1)

    scatter!(axis, sc1[perm], color = :red)
    scatter!(axis, sc2[perm], color = :green)
    scatter!(axis, sc3[perm], color = :blue)

    axis = Axis(figure[2, 2]; ylabel = "running time", xticksvisible = false, xticklabelsvisible = false)

    time1 = dataframe.time[dataframe.label .== LABELS[1]]
    time2 = dataframe.time[dataframe.label .== LABELS[2]]
    time3 = dataframe.time[dataframe.label .== LABELS[3]]

    perm = sortperm(time1)

    plot1 = scatter!(axis, time1[perm], color = :red)
    plot2 = scatter!(axis, time2[perm], color = :green)
    plot3 = scatter!(axis, time3[perm], color = :blue)

    Legend(
        figure[1, 2],
        [plot1, plot2, plot3],
        collect(LABELS),
    )

    save("figure.png", figure)
end

plot(run())
