module TensorBenchmarks

using AbstractTrees
using CliqueTrees
using EinExprs
using OMEinsumContractionOrders
using PythonCall
using SparseArrays
using Yao

using OMEinsum: getixsv, get_size_dict, LeafString
using Yao: TensorNetwork

export make, solve, timecomplexity, spacecomplexity

function make(::Type{TensorNetwork}, circuit)
    n = nqubits(circuit)
    
    network = yao2einsum(circuit;
        initial_state = Dict(zip(1:n, zeros(Int, n))),
        final_state = Dict(zip(1:n, zeros(Int, n))),
        optimizer = nothing,
    )
    
    return network
end

function make(::Type{SizedEinExpr}, circuit)
    n = nqubits(circuit)
    
    network = yao2einsum(circuit;
        initial_state = Dict(zip(1:n, zeros(Int, n))),
        final_state = Dict(zip(1:n, zeros(Int, n))),
        optimizer = nothing,
    )
    
    query = network.code.iy
    indices = network.code.ixs
    tensors = network.tensors
    
    args = map(indices) do i
        return EinExpr(i)
    end
    
    size = get_size_dict(indices, tensors)
    return SizedEinExpr(EinExpr(query, args), size)
end

function make(::Type{Py}, circuit)
    n = nqubits(circuit)
    
    network = yao2einsum(circuit;
        initial_state = Dict(zip(1:n, zeros(Int, n))),
        final_state = Dict(zip(1:n, zeros(Int, n))),
        optimizer = nothing,
    )
    
    query = network.code.iy
    indices = network.code.ixs
    tensors = network.tensors
    
    inputs = pylist(map(i -> pylist(map(Char, i)), indices))
    outputs = pylist(map(Char, query))
    size = pydict(Dict(Char(i) => x for (i, x) in get_size_dict(indices, tensors)))
    return inputs, outputs, size
end

function solve(network::TensorNetwork, optimizer)
    return optimize_code(network, optimizer, MergeVectors())
end

function solve(network::SizedEinExpr, optimizer)
    return einexpr(optimizer, network)
end

function solve(network::Tuple{Py, Py, Py}, optimizer)
    return optimizer.search(network...)
end

function timecomplexity(network::TensorNetwork)
    tree = network.code
    size = get_size_dict(getixsv(tree), network.tensors)
    
    m = 0; index = Dict{Int, Int}(); weights = Float64[]
    
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

function timecomplexity(network)
    return treewidth(decomposition(network)...)
end

function spacecomplexity(network::TensorNetwork)
    tree = network.code
    size = get_size_dict(getixsv(tree), network.tensors)
    maxwidth = 0.0

    for node in PreOrderDFS(tree)
        width = 0.0

        if !isa(node, LeafString)
            for label in node.eins.iy
                width += log2(size[label])
            end
        else
            for string in eachsplit(node.str, "∘")
                label = parse(Int, string)
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
