# CytoFlow

CytoFlow: A Novel Computational Method to Construct Signal Transduction Networks at Single-Cell Resolution based on Flow Networks

## Setup

Clone the repository.

```
git clone https://github.com/aicb-ZhangLabs/CytoFlow.git
```

## Requirements

Install conda environment.

```
conda env create -f cytoflow.yml
```

## Code

Before running the pipelines, modify `load_data` in `utils.py` to load your own data. You need a weighted input network and gene expression profiles to run CytoFlow.

### Standard pipeline

Run `batch_tune.sh` to tune the parameters. You can set your own grid search paradigm.

Run `batch_plot.sh` to plot grid search results and get the optimal parameters.

Run `batch_run.sh` to run the model under the optimal parameters.

### Pairwise flow pattern pipeline

Run `pairwise_flow.py` to get the flow network for every pair of receptors and TFs. You will need to modify the data paths to load your own receptor and TF list.

Run `pairwise_joint_plot.py` to plot the blockwise heatmaps.
