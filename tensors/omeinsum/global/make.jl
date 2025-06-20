include("../../TensorBenchmarks.jl")

using CairoMakie
using CliqueTrees
using DataFrames
using OMEinsum
using OMEinsumContractionOrders
using PythonCall
using Yao

import CSV
import KaHyPar
import Metis

using .TensorBenchmarks

const CTG = pyimport("cotengra")

const LABELS = (
    "CoTenGra",
    "OMEinsum (KaHyPar)",
    "OMEinsum (METIS)",
)

const PAIRS = (
    (Tuple{Py, Py, Py},           () -> CTG.HyperOptimizer()),
    (Tuple{DynamicEinCode, Dict}, () -> HyperND(KaHyParND(); imbalances = 100:10:800)),
    (Tuple{DynamicEinCode, Dict}, () -> HyperND(METISND();   imbalances = 100:10:800)),
)

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

    dir = joinpath("..", "..", "circuits")

    for file in readdir(dir)
        if endswith(file, ".txt")
            path = joinpath(@__DIR__, dir, file)
            query, matrix, weights = TensorBenchmarks.read(path)
                    
            for (label, (T, f)) in zip(LABELS, PAIRS)
                network = make(T, query, matrix, weights)
                
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

    CSV.write("table.csv", dataframe)
    return
end

function plot()
    dataframe = CSV.read("table.csv", DataFrame)

    #----------#
    # raw data #
    #----------#

    figure = Figure(size = (600, 450))

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

    axis = Axis(figure[3, 1]; ylabel = "running time", xticksvisible = false, xticklabelsvisible = false)

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

    save("raw.png", figure)

    #----------------------#
    # preformance profiles #
    #----------------------#

    figure = Figure(size = (600, 450))

    axis = Axis(figure[1, 1]; ylabel = "time complexity")

    (xtc1, xtc2, xtc3), (ytc1, ytc2, ytc3) = profile([tc1 tc2 tc3])

    stairs!(axis, xtc1, ytc1, step=:post, color = :red)
    stairs!(axis, xtc2, ytc2, step=:post, color = :green)
    stairs!(axis, xtc3, ytc3, step=:post, color = :blue)

    axis = Axis(figure[2, 1]; ylabel = "space complexity")

    (xsc1, xsc2, xsc3), (ysc1, ysc2, ysc3) = profile([sc1 sc2 sc3]) 

    stairs!(axis, xsc1, ysc1, step=:post, color = :red)
    stairs!(axis, xsc2, ysc2, step=:post, color = :green)
    stairs!(axis, xsc3, ysc3, step=:post, color = :blue)

    axis = Axis(figure[3, 1]; ylabel = "running time")

    (xtime1, xtime2, xtime3), (ytime1, ytime2, ytime3) = profile([time1 time2 time3])

    plot1 = stairs!(axis, xtime1, ytime1, step=:post, color = :red)
    plot2 = stairs!(axis, xtime2, ytime2, step=:post, color = :green)
    plot3 = stairs!(axis, xtime3, ytime3, step=:post, color = :blue)

    Legend(
        figure[1, 2],
        [plot1, plot2, plot3],
        collect(LABELS),
    )

    save("profile.png", figure)
end

run()
plot()
