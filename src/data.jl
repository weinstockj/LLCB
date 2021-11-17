function ko_summary_data_path()
    path = "/home/users/jweinstk/network_inference/data/experiments/Supplementary_Table_5_RNA_Seq_results.gz"
    if !isfile(path)
        throw(LoadError("path to KO summary results does not exist"))
    end
    return path
end

function ko_rnaseq_data_path()
    path = "/home/users/jweinstk/network_inference/data/experiments/RNA_UMI_dedup_counts.txt"
    if !isfile(path)
        throw(LoadError("path to KO RNA-seq results does not exist"))
    end
    return path
end

function output_dir()
    return "/oak/stanford/groups/pritch/users/jweinstk/network_inference/InferCausalGraph/output/"
end

@memoize function read_ko_summary_data()
    path = ko_summary_data_path()
    @info "$(now()) Reading in summary KO data now"
    df = CSV.read(path, DataFrame)
    rename!(df, Dict("gene_name" => "readout_gene", "sample" => "ko_gene"))
    df.ko_gene = replace.(df.ko_gene, " KO" => "")
    return df
end

@memoize function read_ko_rnaseq_data()
    path = ko_rnaseq_data_path()
    lookup = read_symbol_ensid_lookup()

    df = CSV.read(path, DataFrame)
    @info "$(now()) RNAseq data has $(nrow(df)) rows"
    rename!(df, "Geneid" => "ens_id")
    df.ens_id = first.(split.(df.ens_id, "."))

    df = innerjoin(df, lookup, on = :ens_id)

    @info "$(now()) RNAseq data has $(nrow(df)) rows"

    select!(df, Not([:Chr, :Start, :End, :Strand, :Length]))
    return df
end

@memoize function ko_controls()
    rnaseq = read_ko_rnaseq_data()
    controls = names(rnaseq)[contains.(names(rnaseq), "AAVS1")]
    controls = unique([join(split(x, "_")[3:4], "_") for x in controls]) # convert Donor_1_AAVS1_4 to AAVS1_4
    return controls
end

function ko_targets() 
    summary = read_ko_summary_data()
    non_controls = unique(summary.ko_gene)
    rnaseq = read_ko_rnaseq_data()
    # controls = names(rnaseq)[contains.(names(rnaseq), "AAVS1")]
    # controls = unique([join(split(x, "_")[3:4], "_") for x in controls]) # convert Donor_1_AAVS1_4 to AAVS1_4
    controls = ko_controls()
    return vcat(non_controls, controls)
end

function read_symbol_ensid_lookup()
    summary = read_ko_summary_data()
    lookup = unique(summary[:, ["ens_id", "readout_gene"]])
    return lookup
end

function read_filtered_ko_pairs(pvalue_threshold::Float64 = 5e-25)
    # 118 readouts with 5e-25
    summary = read_ko_summary_data()
    targets = ko_targets()
    filtered = summary[(summary."adj.P.Val" .< pvalue_threshold) .| in.(summary.readout_gene, Ref(targets)), :]

    sort!(filtered, Symbol("adj.P.Val"))

    return filtered
end

function log_and_zscore(x::Vector{Int64})
    x = log10.(Float64.(x) .+ 1.0)
    x = x .- mean(x)
    return x ./ std(x)
end

function normalize_rnaseq(data::DataFrame)
    cols = names(data)

    for c in cols
        if eltype(data[!, c]) <: Number
            transform!(data, Symbol(c) => log_and_zscore, renamecols = false)
        end
    end

    return data
end

function remove_duplicated_readouts(data::DataFrame)

    @info "Prior to removing duplicate readouts, data has $(nrow(data)) rows"

    counts = combine(groupby(data, :readout_gene), nrow)
    unique_genes = counts[counts.nrow .== 1, :readout_gene]

    subset!(data, :readout_gene => (x -> in.(x, Ref(unique_genes))))

    @info "After removing duplicate readouts, data has $(nrow(data)) rows"

    return data
end

function zero_intervened_nodes(data::DataFrame, interventions::Array{String})

    cols = names(data)

    for i in interventions
        for c in cols
            if eltype(data[!, c]) <: Number
                if sum(data.readout_gene .== i) != 1
                    throw("can't find readout gene $i in $c") 
                end
                data[findall(data.readout_gene .== i)[], Symbol(c)] = 0
            end
        end
    end

    return data
end

function read_filtered_rnaseq()
    filtered_pairs = select(read_filtered_ko_pairs(), ["readout_gene", "ko_gene"])
    rnaseq = read_ko_rnaseq_data()

    @info "$(now()) Identified $(nrow(rnaseq)) readouts"

    # targets = unique(filtered_pairs.ko_gene)  # excludes controls
    targets = ko_targets()
    sig_readouts = unique(filtered_pairs.readout_gene)

    @info "$(now()) Identified $(length(sig_readouts)) significant readouts"

    subset!(rnaseq, :readout_gene => (x -> in.(x, Ref(sig_readouts))))

    ko_collection = Dict{String, DataFrame}()

    for ko in targets
        @info "$(now()) ko is $ko"
        # sig_readouts = filtered_pairs[filtered_pairs.ko_gene .== ko, :]
        data = subset_to_interventions(rnaseq, ko)
        data = remove_duplicated_readouts(data)
        if !(ko in ko_controls())
            data = zero_intervened_nodes(data, [ko])
        end
        data = tranpose_df(normalize_rnaseq(data))
        for i in 1:nrow(data)
            data[i, :donor] = ko
        end
        rename!(data, :donor => :intervention)
        # subset!(data, :readout_gene => (x -> in.(x, Ref(sig_readouts.readout_gene))))
        # subset!(data, :readout_gene => (x -> in.(x, Ref(sig_readouts))))
        push!(ko_collection, ko => data)
    end 

    return ko_collection
end

function tranpose_df(data::DataFrame)
    id = "readout_gene"
    cols = setdiff(names(data), [id])
    stacked = DataFrames.stack(data, cols, variable_name = "donor")
    tranposed = DataFrames.unstack(stacked, :donor, Symbol(id), :value)
    return tranposed
end
