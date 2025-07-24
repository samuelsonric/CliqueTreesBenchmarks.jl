include("../../TensorBenchmarks.jl")

using CairoMakie
using CliqueTrees
using DataFrames
using EinExprs
using PythonCall
using Random

import CSV
import KaHyPar
import Metis

using .TensorBenchmarks

const CTG = pyimport("cotengra")

const EXCLUDE = (
    "mc_2020_017.json",
    "mc_2020_062.json",
    "mc_2020_082.json",
    "mc_2022_167.json",
    "wmc_2023_141.json",
)

const LABELS = (
    "CoTenGra",
    "EinExprs (Greedy)",
    "EinExprs (MF)",
)

const PAIRS = (
    (Tuple{Py, Py, Py}, () -> CTG.GreedyOptimizer()),
    (SizedEinExpr,      () -> Greedy()),
    (SizedEinExpr,      () -> LineGraph(MF())),
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

    dir = joinpath("..", "..", "instances")

    for file in readdir(dir)
        if endswith(file, ".json") && file ∉ EXCLUDE
            path = joinpath(@__DIR__, dir, file)
            query, matrix, weights = TensorBenchmarks.read(path)

            any(weights .< 2) && continue

            for (label, (T, f)) in zip(LABELS, PAIRS)
                minscore = (typemax(Float64), typemax(Float64))
                mintime = typemax(Float64)

                for _ in 1:10
                    xperm = randperm(size(matrix, 1)); xinvp = invperm(xperm)
                    yperm = randperm(size(matrix, 2)); yinvp = invperm(yperm)
                    network = make(T, xinvp[query], matrix[xperm, yperm], weights[xperm])
                    
                    optimizer = f()
                    result = solve(network, optimizer)
                    tc = timecomplexity(result)
                    sc = spacecomplexity(result)
                    
                    optimizer = f()
                    time = @elapsed solve(network, optimizer)

                    minscore = min(minscore, (sc, tc))
                    mintime = min(mintime, time)
                end

                minsc, mintc = minscore

                printrow(
                    file,
                    label,
                    mintc,
                    minsc,
                    mintime,
                )
               
                push!(dataframe, (
                    mintc,
                    minsc,
                    mintime,
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

    axis = Axis(figure[1, 1]; ylabel = "time complexity (log)", xticksvisible = false, xticklabelsvisible = false)

    tc1 = dataframe.tc[dataframe.label .== LABELS[1]]
    tc2 = dataframe.tc[dataframe.label .== LABELS[2]]
    tc3 = dataframe.tc[dataframe.label .== LABELS[3]]

    perm = sortperm(tc1)

    scatter!(axis, tc1[perm], color = :red)
    scatter!(axis, tc2[perm], color = :green)
    scatter!(axis, tc3[perm], color = :blue)

    axis = Axis(figure[2, 1]; ylabel = "space complexity (log)", xticksvisible = false, xticklabelsvisible = false)

    sc1 = dataframe.sc[dataframe.label .== LABELS[1]]
    sc2 = dataframe.sc[dataframe.label .== LABELS[2]]
    sc3 = dataframe.sc[dataframe.label .== LABELS[3]]

    perm = sortperm(sc1)

    scatter!(axis, sc1[perm], color = :red)
    scatter!(axis, sc2[perm], color = :green)
    scatter!(axis, sc3[perm], color = :blue)

    axis = Axis(figure[3, 1]; ylabel = "running time (log)", xticksvisible = false, xticklabelsvisible = false)

    time1 = dataframe.time[dataframe.label .== LABELS[1]] .|> log2
    time2 = dataframe.time[dataframe.label .== LABELS[2]] .|> log2
    time3 = dataframe.time[dataframe.label .== LABELS[3]] .|> log2

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

    axis = Axis(figure[1, 1]; ylabel = "time complexity (log)")

    (xtc1, xtc2, xtc3), (ytc1, ytc2, ytc3) = profile([tc1 tc2 tc3])

    stairs!(axis, xtc1, ytc1, step=:post, color = :red)
    stairs!(axis, xtc2, ytc2, step=:post, color = :green)
    stairs!(axis, xtc3, ytc3, step=:post, color = :blue)

    axis = Axis(figure[2, 1]; ylabel = "space complexity (log)")

    (xsc1, xsc2, xsc3), (ysc1, ysc2, ysc3) = profile([sc1 sc2 sc3]) 

    stairs!(axis, xsc1, ysc1, step=:post, color = :red)
    stairs!(axis, xsc2, ysc2, step=:post, color = :green)
    stairs!(axis, xsc3, ysc3, step=:post, color = :blue)

    axis = Axis(figure[3, 1]; ylabel = "running time (log)")

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
