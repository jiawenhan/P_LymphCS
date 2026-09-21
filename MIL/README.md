# MambaMIL

MambaMIL is a multiple instance learning (MIL) training project for whole-slide image (WSI) feature bags. It includes MambaMIL and several commonly used MIL baselines for slide-level classification.

## Supported models

- MambaMIL with Mamba, BiMamba, or SRMamba blocks
- Attention MIL
- CLAM-MB
- TransMIL
- S4MIL
- Mean MIL and Max MIL

## Repository layout

```text
dataset/                 Dataset and split loaders
dataset_csv/             Slide-level metadata
models/                  MIL model definitions
splits/                  Train/validation/test split CSV files
utils/                   Training, evaluation, and loss utilities
mamba/                   Vendored Mamba implementation
main.py                  Classification training entry point
MIL.py                   Per-class metric aggregation utility
Lymph.sh                 Example Lymph classification experiment
```

## Environment

The Mamba kernels require Linux, an NVIDIA GPU, CUDA 11.6 or newer, and a CUDA-compatible PyTorch build. Python 3.10 is recommended. Install PyTorch for the CUDA version on your machine before installing the remaining dependencies.

```bash
conda create -n mambamil python=3.10 -y
conda activate mambamil

# Install the appropriate PyTorch build from https://pytorch.org/get-started/locally/
pip install -r requirements.txt
pip install "causal-conv1d>=1.2.0"
pip install -e ./mamba --no-build-isolation
```

## Data format

This repository trains on pre-extracted patch features rather than raw WSI files. Each slide must have one PyTorch tensor with shape `[num_patches, feature_dim]`.

With `--patch_size 512`, features are resolved as:

```text
<data_root_dir>/pt_files/<backbone>/<slide_id>.pt
```

For other non-empty patch sizes, the expected path is:

```text
<data_root_dir>/<patch_size>/pt_files/<backbone>/<slide_id>.pt
```

The dataset CSV must contain these columns:

| Column | Description |
| --- | --- |
| `case_id` | Patient or case identifier |
| `slide_id` | Slide identifier and feature filename stem |
| `label` | Class label defined by the selected task |
| `dir` | Optional per-slide feature root, used when `--data_root_dir` is omitted |

Split files are named `splits_<fold>.csv` and contain `train`, `val`, and `test` columns populated with slide IDs. The included `dataset_csv/Lymph.csv` and `splits/Lymph_100/splits_0.csv` files provide a concrete example.

## Classification training

The following command trains SR-MambaMIL on the included Lymph metadata and split definition. Replace `/path/to/features` with the directory containing the extracted feature bags.

```bash
python main.py --mil \
  --task Lymph \
  --data_root_dir /path/to/features \
  --split_dir splits/Lymph_100 \
  --results_dir results/Lymph \
  --exp_code mamba_mil_dinov2 \
  --model_type mamba_mil \
  --mambamil_type SRMamba \
  --mambamil_layer 2 \
  --mambamil_rate 5 \
  --backbone dinov2 \
  --patch_size 512 \
  --in_dim 768 \
  --max_epochs 50 \
  --lr 2e-4 \
  --k 1
```

Use `--model_type` to select another architecture. The feature dimension passed through `--in_dim` must match the final dimension of each patch embedding. Add `--early_stopping` to select the checkpoint using validation loss, `--weighted_sample` for class-balanced sampling, and `--log_data` for TensorBoard logs.

`Lymph.sh` is a configurable batch-training example. Environment variables can override its model list, backbone list, GPU, feature root, and output directory.

## Outputs

Each experiment is written to `<results_dir>/<exp_code>_s<seed>/` and contains:

- the resolved configuration in `experiment.txt`;
- the split used for each fold;
- a model checkpoint per fold;
- final validation and test prediction CSV files;
- serialized slide-level prediction dictionaries;
- a fold-level summary CSV.

To aggregate one-vs-rest metrics from the final prediction CSV files:

```bash
python MIL/MIL.py MIL/results/Lymph/mamba_mil_dinov2_s1
```

The command writes `mil_metrics.csv` inside the input directory. Use `--output` to choose another path.

## Adding a dataset

1. Add a metadata CSV under `dataset_csv/` with `case_id`, `slide_id`, `label`, and optionally `dir`.
2. Add one or more split files under `splits/<task>_<label_fraction>/`.
3. Register the task name, label mapping, class count, and CSV path in `MIL/main.py`.
4. Run training with the matching feature root, backbone name, patch size, and input dimension.

Large feature tensors, checkpoints, experiment outputs, and tracking logs are excluded by `.gitignore` and should not be committed to GitHub.

## Acknowledgements

The repository includes code derived from Mamba, CLAM, AttentionDeepMIL, TransMIL, and S4. Review the upstream projects and the vendored `mamba/LICENSE` before redistribution, and cite the corresponding papers when using these models in research.
