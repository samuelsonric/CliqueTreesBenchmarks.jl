module TensorBenchmarks

using AbstractTrees
using CliqueTrees
using EinExprs
using JSON
using OMEinsumContractionOrders
using PythonCall
using SparseArrays

using OMEinsumContractionOrders: NestedEinsum, EinCode, LeafString

export printrow, profile, read, make, solve, timecomplexity, spacecomplexity

function printrow(circuit, algorithm, tc, sc, time)
    print(" | ")
    print(rpad(circuit, 60))
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

function profile(matrix::Matrix{Float64})
    np, ns = size(matrix); npp = np + 1

    ratios = Matrix{Float64}(undef, npp, ns)
    xplots = Vector{Vector{Float64}}(undef, ns)
    yplots = Vector{Vector{Float64}}(undef, ns)

    minima = mapslices(minimum, matrix; dims = 2)
    maxratio = 0.0

    for p in 1:np, s in 1:ns
        ratio = matrix[p, s] / minima[p]
        ratios[p, s] = ratio
        maxratio = max(ratio, maxratio)
    end

    for s in 1:ns
        ratios[npp, s] = 2.0 * maxratio
    end

    sort!(ratios; dims = 1)

    for s in 1:ns
        column = ratios[:, s]; ratio = minimum(column)

        xplot = Float64[]
        yplot = Float64[]

        while ratio < maximum(column)
            index = findlast(column .<= ratio)

            ratio = max(
                column[index],
                column[index + 1],
            )

            push!(xplot, column[index])
            push!(yplot, index / np)
        end

        push!(xplot, column[npp])
        push!(yplot, npp / np)

        xplots[s] = xplot
        yplots[s] = yplot
    end

    return xplots, yplots
end

function read(file::String)
    dict = JSON.parsefile(file)
    einsum = dict["einsum"]

    ixs = Vector{Vector{Int}}(einsum["ixs"])
    iy = Vector{Int}(einsum["iy"])
    size = Dict{String, Int}(dict["size"])

    m = n = 0
    colptr = Int[]; rowval = Int[]; nzval = Int[]
    push!(colptr, 1); p = 1

    for ix in ixs
        for i in ix
            push!(rowval, i); p += 1
            push!(nzval, 1)
            m = max(m, i)
        end

        push!(colptr, p); n += 1
    end

    weights = Vector{Int}(undef, m)

    for (k, v) in size
        i = parse(Int, k); weights[i] = v
    end

    matrix = SparseMatrixCSC(m, n, colptr, rowval, nzval)
    matrix = copy(transpose(copy(transpose(matrix))))
    return iy, matrix, weights
end

function make(::Type{Tuple{EinCode, Dict}}, query, matrix, weights)
    ixs = Vector{Int}[]; iy = Int[]; size_dict = Dict{Int, Int}()

    for j in axes(matrix, 2)
        pstart = matrix.colptr[j]
        pstop = matrix.colptr[j + 1] - 1
        ix = Int[]

        for p in pstart:pstop
            i = matrix.rowval[p]
            push!(ix, i)
        end

        push!(ixs, ix)
    end

    for i in query
        push!(iy, i)
    end

    for (i, v) in enumerate(weights)
        size_dict[i] = v
    end

    return (EinCode(ixs, iy), size_dict)
end

function make(::Type{SizedEinExpr}, query, matrix, weights)
    path = EinExpr(Int[]); size_dict = Dict{Int, Int}()

    for j in axes(matrix, 2)
        pstart = matrix.colptr[j]
        pstop = matrix.colptr[j + 1] - 1
        arg = EinExpr(Int[])

        for p in pstart:pstop
            i = matrix.rowval[p]
            push!(arg.head, i)
        end

        push!(path.args, arg)
    end

    for i in query
        push!(path.head, i)
    end

    for (i, v) in enumerate(weights)
        size_dict[i] = v
    end

    return SizedEinExpr(path, size_dict)
end

function make(::Type{Tuple{Py, Py, Py}}, query, matrix, weights)
    inputs = Py[]; outputs = Char[]; size_dict = Dict{Char, Int}()

    for j in axes(matrix, 2)
        pstart = matrix.colptr[j]
        pstop = matrix.colptr[j + 1] - 1
        input = Char[]

        for p in pstart:pstop
            i = matrix.rowval[p]
            c = Char(i)
            push!(input, c)
        end

        push!(inputs, pylist(input))
    end

    for i in query
        c = Char(i)
        push!(outputs, c)
    end

    for (i, v) in enumerate(weights)
        c = Char(i)
        size_dict[c] = v
    end

    return pylist(inputs), pylist(outputs), pydict(size_dict)
end

function solve(network::Tuple{EinCode, Dict}, optimizer)
    code, size = network
    optcode = optimize_code(code, size, optimizer, MergeVectors())
    return (optcode, size)
end

function solve(network::SizedEinExpr, optimizer)
    return einexpr(optimizer, network)
end

function solve(network::Tuple{Py, Py, Py}, optimizer)
    return optimizer.search(network...)
end

function timecomplexity(network::Tuple{NestedEinsum{L}, Dict{L}}) where {L}
    tree, size = network

    m = 0; index = Dict{L, Int}(); weights = Float64[]
    
    for node in PreOrderDFS(tree)
        if !isa(node, LeafString)
            for label in node.eins.iy
                if !haskey(index, label)
                    index[label] = m += 1
                    push!(weights, log2(size[label]))
                end
            end
        else
            for string in eachsplit(node.str, "∘")
                label = parse(Int, string)
                
                if !haskey(index, label)
                    index[label] = m += 1
                    push!(weights, log2(size[label]))
                end
            end
        end
    end
    
    n = 0; p = 1; colptr = Int[1]; rowval = Int[]; nzval = Int[]
    
    for leaf in Leaves(tree)
        for string in eachsplit(leaf.str, "∘")
            label = parse(Int, string)
            i = index[label]
            p += 1; push!(rowval, i); push!(nzval, 1)
        end
        
        n += 1; push!(colptr, p)
    end
    
    hypergraph = SparseMatrixCSC(m, n, colptr, rowval, nzval)
    hypergraph = copy(transpose(hypergraph))
    dualgraph = hypergraph' * hypergraph
    return treewidth(weights, dualgraph; alg=m:-1:1)
end

function timecomplexity(network::SizedEinExpr{L}) where {L}
    tree = network.path
    size = network.size

    m = 0; index = Dict{L, Int}(); weights = Float64[]
    
    for node in PreOrderDFS(tree)
        for label in head(node)
            if !haskey(index, label)
                index[label] = m += 1
                push!(weights, log2(size[label]))
            end
        end
    end
    
    n = 0; p = 1; colptr = Int[1]; rowval = Int[]; nzval = Int[]
    
    for leaf in Leaves(tree)
        for label in head(leaf)
            i = index[label]
            p += 1; push!(rowval, i); push!(nzval, 1)
        end
        
        n += 1; push!(colptr, p)
    end
        
    hypergraph = SparseMatrixCSC(m, n, colptr, rowval, nzval)
    hypergraph = copy(transpose(hypergraph))
    dualgraph = hypergraph' * hypergraph
    return treewidth(weights, dualgraph; alg=m:-1:1)
end

function timecomplexity(network::Py)
    tree = network
    size = network.size_dict

    m = 0; index = Dict{Char, Int}(); weights = Float64[]

    for (node, _, _) in tree.descend()
        labels = tree.get_legs(node)

        for label in labels
            label = pyconvert(Char, label)

            if !haskey(index, label)
                index[label] = m += 1
                push!(weights, log2(pyconvert(Int, size[label])))
            end
        end
    end

    for leaf in tree.gen_leaves()
        labels = tree.get_legs(leaf)

        for label in labels
            label = pyconvert(Char, label)

            if !haskey(index, label)
                index[label] = m += 1
                push!(weights, log2(pyconvert(Int, size[label])))
            end
        end
    end

    n = 0; p = 1; colptr = Int[1]; rowval = Int[]; nzval = Int[]

    for leaf in tree.gen_leaves()
        labels = tree.get_legs(leaf)

        for label in labels
            label = pyconvert(Char, label)
            i = index[label]
            p += 1; push!(rowval, i); push!(nzval, 1)
        end

        n += 1; push!(colptr, p)
    end

    hypergraph = SparseMatrixCSC(m, n, colptr, rowval, nzval)
    hypergraph = copy(transpose(hypergraph))
    dualgraph = hypergraph' * hypergraph
    return treewidth(weights, dualgraph; alg=m:-1:1)
end

function spacecomplexity(network::Tuple{NestedEinsum{L}, Dict{L}}) where {L}
    tree, size = network
    maxwidth = 0.0

    for node in PreOrderDFS(tree)
        width = 0.0

        if !isa(node, LeafString)
            for label in node.eins.iy
                width += log2(size[label])
            end
        else
            for string in eachsplit(node.str, "∘")
                label = parse(L, string)
                width += log2(size[label])
            end
        end

        maxwidth = max(width, maxwidth)
    end

    return maxwidth
end

function spacecomplexity(network::SizedEinExpr)
    tree = network.path
    size = network.size
    maxwidth = 0.0
    
    for node in PreOrderDFS(tree)
        width = 0.0
        
        for label in head(node)
            width += log2(size[label])
        end
        
        maxwidth = max(width, maxwidth)
    end
    
    return maxwidth
end

function spacecomplexity(network::Py)
    return pyconvert(Float64, network.contraction_width())
end

end
