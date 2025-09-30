include("../../TensorBenchmarks.jl")

using CairoMakie
using CliqueTrees
using DataFrames
using OMEinsumContractionOrders
using PythonCall

using OMEinsumContractionOrders: EinCode

import CSV
import KaHyPar
import Metis

using .TensorBenchmarks

const CTG = pyimport("cotengra")

const EXCLUDE = (
    "mc_2020_017.json",
    "mc_2020_062.json",
    "mc_2020_082.json",
    "mc_2022_087.json",
    "mc_2022_167.json",
    "mc_2023_002.json",
    "mc_2023_151.json",
    "mc_2023_188.json",
    "mc_2023_arjun_117.json",
    "mc_rw_32.sk_4_38.json",
    "mc_rw_c7552.isc.json",
    "wmc_2021_061.json",
    "wmc_2021_145.json",
    "wmc_2023_141.json",
    "wmc_2023_152.json",
    "qc_maxcut_n18_r17_p100.json",
)

const LABELS = (
    "CoTenGra",
    "OMEinsum (KaHyPar)",
    "OMEinsum (METIS)",
)

const PAIRS = (
    (Tuple{Py, Py, Py},    () -> CTG.HyperOptimizer()),
    (Tuple{EinCode, Dict}, () -> HyperND(; dis = KaHyParND(), width = 50, imbalances = 100:10:800)),
    (Tuple{EinCode, Dict}, () -> HyperND(; dis = METISND(),   width = 50, imbalances = 100:10:800)),
)

function run()
    if "table.csv" in readdir(".")
        dataframe = CSV.read("table.csv", DataFrame; types=Dict(
            :tc => Float64,
            :sc => Float64,
            :time => Float64,
            :file => String,
            :label => String,
        ))
    else
        dataframe = DataFrame(
            tc=Float64[],    # time complexity
            sc=Float64[],    # space complexity
            time=Float64[],  # run time
            file = String[], # circuit file
            label=String[],  # label
        )
    end

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
                any(dataframe.label .== label .&& dataframe.file .== file) && continue
                network = make(T, query, matrix, weights)
                tc = sc = time = typemax(Float64)

                try
                    optimizer = f()
                    result = solve(network, optimizer)
                    tc = timecomplexity(result)
                    sc = spacecomplexity(result)
                
                    optimizer = f()
                    time = @elapsed solve(network, optimizer)
                catch
                end

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

                CSV.write("table.csv", dataframe)
            end
        end
    end

    return
end

function plot()
    dataframe = CSV.read("table.csv", DataFrame)
    n = length(dataframe.tc[dataframe.label .== LABELS[1]])

    #----------#
    # raw data #
    #----------#

    figure = Figure(size = (600, 750))

    tc1 = 2 .^ dataframe.tc[dataframe.label .== LABELS[1]]
    tc2 = 2 .^ dataframe.tc[dataframe.label .== LABELS[2]]
    tc3 = 2 .^ dataframe.tc[dataframe.label .== LABELS[3]]

    tcg2 = trunc(Int, sum(tc2 .> tc1) / n * 100.0)
    tcl2 = trunc(Int, sum(tc2 .< tc1) / n * 100.0)
    tcg3 = trunc(Int, sum(tc3 .> tc1) / n * 100.0)
    tcl3 = trunc(Int, sum(tc3 .< tc1) / n * 100.0)

    axis = Axis(figure[1, 1]; xscale = log2, yscale = log2, title = "time complexity", xlabel = LABELS[2], ylabel = LABELS[1])
    xlims!(axis, 2.0^10, 2.0^70)
    ylims!(axis, 2.0^10, 2.0^70)
    scatter!(axis, tc2, tc1; color = :red)
    lines!(axis, [2.0^10, 2.0^70], [2.0^10, 2.0^70]; color = :black)
    text!(2.0^60, 2.0^20; text = "$tcg2%", color = :black, align = (:center, :center))
    text!(2.0^20, 2.0^60; text = "$tcl2%", color = :black, align = (:center, :center))

    axis = Axis(figure[1, 2]; xscale = log2, yscale = log2, title = "time complexity", xlabel = LABELS[3])
    xlims!(axis, 2.0^10, 2.0^70)
    ylims!(axis, 2.0^10, 2.0^70)
    scatter!(axis, tc3, tc1; color = :blue)
    lines!(axis, [2.0^10, 2.0^70], [2.0^10, 2.0^70]; color = :black)
    text!(2.0^60, 2.0^20; text = "$tcg3%", color = :black, align = (:center, :center))
    text!(2.0^20, 2.0^60; text = "$tcl3%", color = :black, align = (:center, :center))

    sc1 = 2 .^ dataframe.sc[dataframe.label .== LABELS[1]]
    sc2 = 2 .^ dataframe.sc[dataframe.label .== LABELS[2]]
    sc3 = 2 .^ dataframe.sc[dataframe.label .== LABELS[3]]

    scg2 = trunc(Int, sum(sc2 .> sc1) / n * 100.0)
    scl2 = trunc(Int, sum(sc2 .< sc1) / n * 100.0)
    scg3 = trunc(Int, sum(sc3 .> sc1) / n * 100.0)
    scl3 = trunc(Int, sum(sc3 .< sc1) / n * 100.0)

    axis = Axis(figure[2, 1]; xscale = log2, yscale = log2, title = "space complexity", xlabel = LABELS[2], ylabel = LABELS[1])
    xlims!(axis, 2.0^10, 2.0^60)
    ylims!(axis, 2.0^10, 2.0^60)
    scatter!(axis, sc2, sc1; color = :red)
    lines!(axis, [2.0^10, 2.0^60], [2.0^10, 2.0^60]; color = :black)
    text!(2.0^50, 2.0^20; text = "$scg2%", color = :black, align = (:center, :center))
    text!(2.0^20, 2.0^50; text = "$scl2%", color = :black, align = (:center, :center))

    axis = Axis(figure[2, 2]; xscale = log2, yscale = log2, title = "space complexity", xlabel = LABELS[3])
    xlims!(axis, 2.0^10, 2.0^60)
    ylims!(axis, 2.0^10, 2.0^60)
    scatter!(axis, sc3, sc1; color = :blue)
    lines!(axis, [2.0^10, 2.0^60], [2.0^10, 2.0^60]; color = :black)
    text!(2.0^50, 2.0^20; text = "$scg3%", color = :black, align = (:center, :center))
    text!(2.0^20, 2.0^50; text = "$scl3%", color = :black, align = (:center, :center))

    time1 = dataframe.time[dataframe.label .== LABELS[1]]
    time2 = dataframe.time[dataframe.label .== LABELS[2]]
    time3 = dataframe.time[dataframe.label .== LABELS[3]]

    timeg2 = trunc(Int, sum(time2 .> time1) / n * 100.0)
    timel2 = trunc(Int, sum(time2 .< time1) / n * 100.0)
    timeg3 = trunc(Int, sum(time3 .> time1) / n * 100.0)
    timel3 = trunc(Int, sum(time3 .< time1) / n * 100.0)

    axis = Axis(figure[3, 1]; xscale = log2, yscale = log2, title = "running time", xlabel = LABELS[2], ylabel = LABELS[1])
    xlims!(axis, 2.0^-5, 2.0^10)
    ylims!(axis, 2.0^-5, 2.0^10)
    scatter!(axis, time2, time1; color = :red)
    lines!(axis, [2.0^-5, 2.0^10], [2.0^-5, 2.0^10]; color = :black)
    text!(2.0^7, 2.0^-2; text = "$timeg2%", color = :black, align = (:center, :center))
    text!(2.0^-2, 2.0^7; text = "$timel2%", color = :black, align = (:center, :center))

    axis = Axis(figure[3, 2]; xscale = log2, yscale = log2, title = "running time", xlabel = LABELS[3])
    xlims!(axis, 2.0^-5, 2.0^10)
    ylims!(axis, 2.0^-5, 2.0^10)
    scatter!(axis, time3, time1; color = :blue)
    lines!(axis, [2.0^-5, 2.0^10], [2.0^-5, 2.0^10]; color = :black)
    text!(2.0^7, 2.0^-2; text = "$timeg3%", color = :black, align = (:center, :center))
    text!(2.0^-2, 2.0^7; text = "$timel3%", color = :black, align = (:center, :center))

    save("raw.png", figure)

    #----------------------#
    # preformance profiles #
    #----------------------#

    figure = Figure(size = (500, 750))

    axis = Axis(figure[1, 1]; xscale = log2, title = "time complexity", xlabel = "performance ratio", ylabel = "problems solved")
    xlims!(axis, 1.0, 2.0^12)
    ylims!(axis, 0.0, 1.0)

    (xtc1, xtc2, xtc3), (ytc1, ytc2, ytc3) = profile([tc1 tc2 tc3])

    stairs!(axis, xtc1, ytc1; step=:post, color = :red)
    stairs!(axis, xtc2, ytc2; step=:post, color = :green)
    stairs!(axis, xtc3, ytc3; step=:post, color = :blue)

    axis = Axis(figure[2, 1]; xscale = log2, title = "space complexity", xlabel = "performance ratio", ylabel = "problems solved")
    xlims!(axis, 1.0, 2.0^12)
    ylims!(axis, 0.0, 1.0)

    (xsc1, xsc2, xsc3), (ysc1, ysc2, ysc3) = profile([sc1 sc2 sc3]) 

    stairs!(axis, xsc1, ysc1; step=:post, color = :red)
    stairs!(axis, xsc2, ysc2; step=:post, color = :green)
    stairs!(axis, xsc3, ysc3; step=:post, color = :blue)

    axis = Axis(figure[3, 1]; xscale = log2, title = "running time", xlabel = "performance ratio", ylabel = "problems solved")
    xlims!(axis, 1.0, 2.0^6)
    ylims!(axis, 0.0, 1.0) 

    (xtime1, xtime2, xtime3), (ytime1, ytime2, ytime3) = profile([time1 time2 time3])

    plot1 = stairs!(axis, xtime1, ytime1; step=:post, color = :red)
    plot2 = stairs!(axis, xtime2, ytime2; step=:post, color = :green)
    plot3 = stairs!(axis, xtime3, ytime3; step=:post, color = :blue)

    Legend(
        figure[1, 2],
        [plot1, plot2, plot3],
        collect(LABELS),
    )

    save("profile.png", figure)
end

run()
plot()
