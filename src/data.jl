function ko_summary_data_path()
    # path = "/home/users/jweinstk/network_inference/data/experiments/Supplementary_Table_5_RNA_Seq_results.gz"
    path = "/oak/stanford/groups/pritch/users/jweinstk/perturbation_data/rnaseq_pipeline/scripts/output/diffeq/txt/differential_expression_results_regressed_pcs.tsv"
    if !isfile(path)
        error("path to KO summary results does not exist")
    end
    return path
end

function ko_rnaseq_data_path()
    # path = "/home/users/jweinstk/network_inference/data/experiments/RNA_UMI_dedup_counts.txt"
    path = "/oak/stanford/groups/pritch/users/jweinstk/perturbation_data/rnaseq_pipeline/scripts/output/diffeq/txt/vst_normalized_counts_transpose.tsv"
    if !isfile(path)
        error("path to KO RNA-seq results does not exist")
    end
    return path
end

function covariates_path()
    path = "/oak/stanford/groups/pritch/users/jweinstk/perturbation_data/rnaseq_pipeline/scripts/output/diffeq/txt/covariates.tsv"

    if !isfile(path)
        error("path to covariates does not exist")
    end
    return path
end

function simulated_data_dir()
    dir = "/oak/stanford/groups/pritch/users/jweinstk/network_inference/ground_truth_simulation/output"
    return dir
end

function simulated_data_paths()
    files = readdir(simulated_data_dir(), join = true)
    pattern = "expression_simulation"

    return filter(x -> occursin(pattern, x), files)
end

function output_dir()
    return "/oak/stanford/groups/pritch/users/jweinstk/network_inference/InferCausalGraph/output/"
end

@memoize function read_ko_summary_data(path = ko_summary_data_path())
    # path = ko_summary_data_path()
    @info "$(now()) Reading in summary KO data now"
    df = CSV.read(path, DataFrame; delim = "\t")
    df = mapcols(collect, df)
    rename!(
        df, 
        Dict("gene_name" => "readout_gene", "KO" => "ko_gene")
    )
    df.ko_gene = replace.(df.ko_gene, " KO" => "")
    return df
end

@memoize function read_normalized_counts(path = ko_rnaseq_data_path())
    lookup = read_symbol_ensid_lookup()

    df = CSV.read(path, DataFrame; delim = "\t")
    df = mapcols(collect, df)
    @info "$(now()) RNAseq data has $(nrow(df)) rows"

    df = innerjoin(df, lookup, on = :gene_id)

    @info "$(now()) RNAseq data has $(nrow(df)) rows after merging"

    return df
end

@memoize function read_covariates(path = covariates_path())

    df = CSV.read(path, DataFrame; delim = "\t")
    df = mapcols(collect, df)

    @info "$(now()) covariates has $(nrow(df)) rows"

    return df
end

@memoize function read_metadata(path = "/oak/stanford/groups/pritch/users/jweinstk/perturbation_data/metadata/sample_meta_data_2022_09_29.tsv")

    @info "$(now()) reading in experiment metadata"
    df = CSV.read(path, DataFrame)
    # donors are only unique within experiment version; concatenate for convenience
    df.Donor = string.(df.Donor, "_", df.experiment_version)
    return df
end

@memoize function ko_controls()
    # rnaseq = read_normalized_counts()
    # controls = names(rnaseq)[contains.(names(rnaseq), "AAVS1")]
    # controls = unique([join(split(x, "_")[3:4], "_") for x in controls]) # convert Donor_1_AAVS1_4 to AAVS1_4
    metadata = read_metadata()
    controls = unique(metadata.KO[metadata.is_control])
    return controls
end

@memoize function ko_targets() 
    summary = read_ko_summary_data()
    non_controls = unique(summary.ko_gene)
    rnaseq = read_normalized_counts()
    controls = ko_controls()
    return vcat(non_controls, controls)
end

@memoize function read_symbol_ensid_lookup(path = "/oak/stanford/groups/pritch/users/jweinstk/perturbation_data/rnaseq_pipeline/scripts/output/diffeq/txt/txdb.tsv")
    # summary = read_ko_summary_data()
    # lookup = unique(summary[:, ["gene_id", "readout_gene"]])
    
    lookup = CSV.read(path, DataFrame; delim = "\t")
    rename!(
        lookup, 
        Dict("gene_name" => "readout_gene")
    )
    return lookup
end

function read_filtered_ko_pairs(pvalue_threshold::Float64 = 5e-25)
    # 118 readouts with 5e-25
    summary = read_ko_summary_data()
    targets = ko_targets()
    # filtered = summary[(summary."padj" .< pvalue_threshold) .| in.(summary.readout_gene, Ref(targets)), :]
    filtered = summary[(summary."pvalue" .< pvalue_threshold) .| in.(summary.readout_gene, Ref(targets)), :]

    # sort!(filtered, Symbol("padj"))
    sort!(filtered, Symbol("pvalue"))

    return filtered
end

function log_and_zscore(x::Vector{Int64})
    x = log10.(Float64.(x) .+ 1.0)
    x = x .- mean(x)
    return x ./ std(x)
end

function log_and_zscore(x::Vector{Float64})
    x = log10.(x .+ 1.0)
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

function read_filtered_normalized_rnaseq()
    filtered_pairs = select(read_filtered_ko_pairs(), ["readout_gene", "ko_gene"])
    rnaseq = read_normalized_counts()
    metadata = read_metadata()
    covariates = read_covariates()
    active_samples = setdiff(names(rnaseq), ["gene_id", "readout_gene"])
    subset!(metadata, :Sample => (x -> in.(x, Ref(active_samples))))

    @info "$(now()) Identified $(nrow(rnaseq)) readouts"

    targets = ko_targets()
    sig_readouts = unique(filtered_pairs.readout_gene)

    @info "$(now()) Identified $(length(sig_readouts)) readouts after filtering"

    subset!(rnaseq, :readout_gene => (x -> in.(x, Ref(sig_readouts))))

    rnaseq = regress_out_covariates(rnaseq, covariates)

    ko_collection = DataFrame()

    for ko in targets
        @info "$(now()) ko is $ko"
        data = copy(subset_to_interventions(rnaseq, metadata, ko))
        data = remove_duplicated_readouts(data)
        if !(ko in ko_controls())
               data = zero_intervened_nodes(data, [ko])
        end
        data = tranpose_df(data)
        println("ko = $ko has $(nrow(data)) rows")
        data[!, :intervention] .= ko
        data[!, :donor] .= replace.(data[!, :donor], "_$(ko)" => "")
        data[!, :donor] .= replace.(data[!, :donor], "Donor_" => "")
        for i in 1:nrow(data)
            push!(ko_collection, data[i, :])
        end
    end 

    return ko_collection[:, vcat(["donor", "intervention"], setdiff(targets, ko_controls()))]
end

function tranpose_df(data::DataFrame)
    id = "readout_gene"
    cols = setdiff(names(data), [id])
    stacked = DataFrames.stack(data, cols, variable_name = "donor")
    tranposed = DataFrames.unstack(stacked, :donor, Symbol(id), :value)
    return tranposed
end

function regress_out_covariates(rnaseq::DataFrame, covariates::DataFrame)
    data = select(rnaseq, findall(col -> eltype(col) <: AbstractFloat, eachcol(rnaseq)))
    covars = select(covariates, findall(col -> eltype(col) <: AbstractFloat, eachcol(covariates)))

    if !(ncol(data) == nrow(covars)) 
        error("expression data and covariate dimensions don't match")
    end

    mat = Matrix(data)
    covars = Matrix(covars)
    covars = [ones(size(covars, 1)) covars]
    tmat = transpose(mat)
    newmat = copy(tmat)

    @inbounds @simd for j in 1:size(tmat, 2)
        y = view(tmat, :, j)
        β = covars \ y
        ypred = covars * β
        R2 = round(cor(ypred, y) ^ 2; digits = 3)
        println("Rsq = $R2 for col $j with covariates")
        if R2 > .75
            println("β = $β")
        end
        resid = y - ypred
        newmat[:, j] .= resid .+ mean(y)
    end

    tnewmat = Matrix(transpose(newmat))
    
    newdata = DataFrame(tnewmat, :auto)
    rename!(newdata, names(data))
    newdata.gene_id = rnaseq.gene_id
    newdata.readout_gene = rnaseq.readout_gene

    return newdata
end
