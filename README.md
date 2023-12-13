# InferCausalGraph
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://weinstockj.github.io/LLCB/dev)
[![CI](https://github.com/weinstockj/LLCB/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/weinstockj/LLCB/actions/workflows/ci.yml)

This repo contains the implementation of the 'LLCB' method, which is described
in Weinstock & Arce et al. 


## Quick-start

Assuming that you have expression data properly formatted into a dataframe called `expression`:

```julia
using InferCausalGraph
const graph = interventionGraph(expression)
const model_pars = get_model_params(false, 1.0, 0.01)
const sampling_pars = get_sampling_params(false)
model = fit_cyclic_model(graph, false, model_pars, sampling_pars)
edges = get_cyclic_matrices(graph, false)[3]
parsed_chain = parse_cyclic_chain(model[1], model[2], edges)
```

## Installation

Currently using Julia 1.9.4. Please see `mannifest.toml` for more details on 
which Julia modules are required. Later versions of Julia are likely to work
but have not been tested. 

To install, you can clone this repo, activate the project, modify `LOAD_PATH` (as needed)
and run:

```julia
using Pkg
Pkg.instantiate() # only need to this run once to setup the package
using InferCausalGraph # on my machine, takes 13.68 seconds in Julia 1.9.4
```

## Input data formatting

The input variable `expression` should be a dataframe where the rows
indicate the sample. Coluns should include the identity of the 
sample `donor`, the intervention performed `intervention`, and then additional
column for each readout gene, where the column names are the gene symbols. 
Overall, the dataframe should have one row per sample, and P + 2 columns, where
P is the number of readout genes. 


For an example of what this data might look like, run the following example 
from the test script `test/extended_test.jl`

```julia
include("test/extended_test.jl")
sim_cyclic_expression(cyclic_chain_graph(), 3, 50, true)
```

This will simulate expression data for a 5 gene graph. 
Each KO has been performed in 3 'donors' with 50
replicates (yes, a somewhat optimistic setting). Control data (no KO's) are also included. 

The first ten rows of the data looks like:

```
 Row │ gene_1   gene_2   gene_3   gene_4   gene_5   intervention  donor
     │ Float64  Float64  Float64  Float64  Float64  String        String
─────┼───────────────────────────────────────────────────────────────────
   1 │ 0.0      3.36514  4.67425  4.92861  5.44028  gene_1        1
   2 │ 0.0      2.63952  3.93181  4.63513  4.89527  gene_1        1
   3 │ 0.0      2.9394   4.14463  4.9676   5.48139  gene_1        1
   4 │ 0.0      3.23764  4.36584  4.67192  5.13055  gene_1        1
   5 │ 0.0      2.86171  4.43841  4.50357  4.76188  gene_1        1
   6 │ 0.0      2.66913  3.957    4.81993  5.05754  gene_1        1
   7 │ 0.0      2.58999  3.68238  4.23696  5.25393  gene_1        1
   8 │ 0.0      2.2197   3.82162  4.67875  4.75546  gene_1        1
   9 │ 0.0      2.89671  4.33153  4.52459  4.74094  gene_1        1
  10 │ 0.0      2.72936  4.18565  4.70949  4.98802  gene_1        1
```

## Input data notes

In our manuscript, we used bulk RNA-seq data as the read-out. Our processing
pipeline is available [here](https://github.com/weinstockj/RNAseq-perturbation-CD4-pipeline). 

Please note that we analyzed the DESeq2 variance stabilizing transform (`vst`) function to 
normalize the data. We also recommend estimating expression PCs and regressing out those
which correspond to unwanted global sources of variation. 

## Contact
Please contact Josh Weinstock <jweins17@jhu.edu> with questions. 

## Citation
If you use this method, please cite following manuscript:

https://www.biorxiv.org/content/10.1101/2023.09.17.557749v2
