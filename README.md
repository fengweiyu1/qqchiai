# Weight level fault localization for neural networks

This repository provides the implementation and experimental code for weight-level fault localization in neural networks. The experiments use Fashion-MNIST, CIFAR-10, and GTSRB, and cover faulty model construction, fault localization, and neural-network repair. Fault injection and repair operate on the final linear layer, `fc2.weight`.

## System Requirements

- Python: 3.9
- PyTorch: 1.13.1
- torchvision: 0.14.1
- NumPy: 1.24.3
- Pillow: 11.2.1
- SciPy: 1.13.1
- Matplotlib: 3.9.4

The scripts use CUDA when available and CPU otherwise. Dependency versions are specified in `requirements.txt`.

## Installation

From the repository root, create a virtual environment and install the dependencies:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

To check the pipeline using synthetic inputs:

```bash
python scripts/smoke_test.py
```

## Preparation

### Datasets

The experiments use the following datasets and configuration files:

| Dataset | Download | Local directory | Configuration |
| --- | --- | --- | --- |
| Fashion-MNIST | [Official repository](https://github.com/zalandoresearch/fashion-mnist#get-the-data) | `data/fashionmnist` | `configs/fm.json` |
| CIFAR-10 | [Official dataset page](https://www.cs.toronto.edu/~kriz/cifar.html) | `data/cifar10` | `configs/cifar10.json` |
| GTSRB | [Download information](https://github.com/houbensebastian/GermanTrafficSignBenchmarks#gtsrb--german-traffic-sign-recognition-benchmark) | `data/gtsrb` | `configs/gtsrb.json` |

Missing dataset files are downloaded automatically when first accessed. Existing data is reused. For manual installation, see [data/README.md](data/README.md).

The commands below use Fashion-MNIST. Replace `configs/fm.json` with `configs/cifar10.json` or `configs/gtsrb.json` to run the other datasets.

### Model Training

Train the benchmark model:

```bash
python scripts/train_model.py --config configs/fm.json
```

The checkpoint is saved to `outputs/fm/models/fm_arachne_rq1_best_model.pth`. To use an existing compatible checkpoint, set `model_path` in the configuration or pass `--model-path` to the faulty model generation command. Pretrained checkpoints are not included in the repository; retraining produces a new benchmark instance.

### Construction of Benchmarks

Generate single-fault models, then construct multiple-fault models from the accepted single faults:

```bash
python scripts/generate_mutants.py --config configs/fm.json --mode single
python scripts/generate_mutants.py --config configs/fm.json --mode multi
```

By default, generation requests 200 single-fault models and up to 50 models for each multiple-fault level: 2, 3, 4, and 5 faults. The accepted count depends on the generation criteria.

The resulting checkpoints and `mutant_pool.json` files are stored under:

```text
outputs/fm/mutants_single/
outputs/fm/mutants_multi/
```

## Usage

Experiment settings are stored in `configs/`. Command-line arguments override the corresponding configuration values. Each script supports `--help` to list its options and `--dry-run` to display the resolved settings without loading data or running an experiment.

```bash
python scripts/run_localization.py --config configs/fm.json --dry-run
```

Use `--mode single` for the single-fault pool and `--mode multi` for the multiple-fault pool. Use `--data-dir` to specify another dataset location, `--mutant-pool` to select an existing pool for localization or repair, and `--output-dir` to choose a result directory. Relative paths are resolved from the repository root.

The analysis entry points are `analysis/rq1.py`, `analysis/rq2.py`, and `analysis/rq3.py`. They read the JSON outputs for the corresponding RQ and save PNG/PDF figures under `figures/`, CSV tables under `tables/`, and analysis metadata in `rq<N>_analysis.json`. They process both fault modes by default; use `--mode single` or `--mode multi` to process one mode. Use `--results-dir` to point to another RQ result root and `--output-dir` to save the analysis elsewhere.

The commands in the following section use separate result directories for each RQ. Each run saves its effective settings in `run_settings.json`. To repeat an experiment, choose a new output directory; scripts refuse to overwrite an existing primary result file.

## Reproduction of Experimental Results

### RQ1: Which Combination of Our Approach Performs Best?

RQ1 compares the ten configurations in Section 4, formed by combining two involvement measures with five result-side measures. Run gradient involvement and forward-impact involvement on both fault pools:

```bash
python scripts/run_localization.py --config configs/fm.json --mode single \
    --baseline methods --execution both --output-dir outputs/fm/rq1/single
python scripts/run_localization.py --config configs/fm.json --mode multi \
    --baseline methods --execution both --output-dir outputs/fm/rq1/multi
```

The faulty-weight ranks are saved to:

```text
outputs/fm/rq1/single/methods/faulty_ranks.json
outputs/fm/rq1/multi/methods/faulty_ranks.json
```

Result keys use `G` for gradient involvement and `FI` for forward-impact involvement. The runner also records auxiliary configurations. The RQ1 analysis selects exactly the ten combinations of `G`/`FI` with `QQ`, `QQ_deficit`, `QQ_loss`, `QQ_cosSim`, and `QQ_max_log`. Ranks are 1-based.

Generate the RQ1 curves, normalized AUC summaries, and pairwise comparison matrices:

```bash
python analysis/rq1.py --config configs/fm.json
```

The default inspected-weight windows are 1% and 20%; use `--windows` to change them. Pairwise comparisons use mutant-level AUCs, two-sided Wilcoxon tests, and Holm correction over all 45 pairs per mode, window, and metric.

The selected configuration, `G_Cos`, is called **our method** in RQ2 and RQ3. Its implementation key is `G_QQ_cosSim_cos_ochiai`.

### RQ2: How Does Our Method Compare with Existing Methods?

RQ2 compares our method with the existing methods and their weight-level adaptations. Run the localization methods and the VeRe baseline on both pools:

```bash
python scripts/run_localization.py --config configs/fm.json --mode single \
    --baseline all --execution both --output-dir outputs/fm/rq2/single
python scripts/run_localization.py --config configs/fm.json --mode multi \
    --baseline all --execution both --output-dir outputs/fm/rq2/multi
```

For each mode, results are saved under `outputs/fm/rq2/<mode>/methods/faulty_ranks.json` and `outputs/fm/rq2/<mode>/vere/faulty_ranks.json`. The following keys identify the comparison methods:

| Method | Result group | Key |
| --- | --- | --- |
| Our method (`G_Cos`) | `methods` | `G_QQ_cosSim_cos_ochiai` |
| ArachneVar | `methods` | `G_QB_cos_ochiai` |
| DeepFaultVar | `methods` | `G_DFVar_k10` |
| GuiderVar | `methods` | `G_GDVar_k10` |
| DeepFault | `methods` | `BB_wfn` |
| Guider | `methods` | `BQ_wfn` |
| VeRe | `vere` | `VeRe_it10`, `VeRe_it15`, or `VeRe_it20` |

Generate the RQ2 curves, normalized AUC summaries, and pairwise comparisons:

```bash
python analysis/rq2.py --config configs/fm.json
```

The analysis aligns methods and VeRe by mutant ID. Following the final experiment plotting code, it selects the VeRe interval multiplier with the highest mean DR through 20% of weights from 1.0, 1.5, and 2.0, separately for each fault mode and dataset. The selected key and excluded IDs are recorded in `rq2_analysis.json`. Use `--vere-selection it10`, `it15`, or `it20` to choose a fixed multiplier. Pairwise comparisons apply Holm correction over all 21 pairs per mode, window, and metric. Implementation details and adaptations are described in [THIRD_PARTY.md](THIRD_PARTY.md).

### RQ3: Does More Accurate Localisation Lead to Better Repair?

RQ3 compares the repair outcomes obtained by selecting weights with different localization methods. All methods use a shared differential-evolution repair procedure:

```bash
python scripts/run_repair.py --config configs/fm.json --mode single \
    --output-dir outputs/fm/rq3/single
python scripts/run_repair.py --config configs/fm.json --mode multi \
    --output-dir outputs/fm/rq3/multi
```

The default comparison includes `ours` (`G_Cos`), `arachnevar`, `deepfaultvar`, `guidervar`, `deepfault`, `guider`, and `vere`, along with `random` and the oracle control `truefault`. Use `--localizers` to select a comma-separated subset.

Repair recomputes localization on one half of the test split and evaluates on the held-out half. The default repair budgets are 1, 2, 3, 5, and 8 for single faults, and 2, 5, and 10 for multiple faults. Both modes use alpha values 1 and 10, a DE population of 100, and at most 100 generations. These settings can be changed in the configuration or through command-line arguments.

Results are saved to:

```text
outputs/fm/rq3/single/repair_results.json
outputs/fm/rq3/multi/repair_results.json
```

Each result records the method (`localizer`), repair budget (`n_fix`), alpha, and number of selected injected faults (`fault_hit`). Results under `R` describe the repair split; results under `E` describe the held-out split. Compare methods at the same repair budget and alpha using:

- Repair rate: `n_fixed / n_broken`.
- Breakage rate: `n_new_broken / n_mut_correct`.

For aggregate rates, sum the corresponding counts across mutants before taking the ratio. A rate is undefined when its denominator is zero.

Generate the RQ3 repair and breakage plots and statistical tables:

```bash
python analysis/rq3.py --config configs/fm.json
```

`tables/rq3_summary.csv` includes pooled and per-mutant mean rates, fault-hit rates, and accuracy ratios. `tables/rq3_comparisons.csv` compares our method with each available baseline using paired mutant-level repair rates, with Holm correction within each mode, budget, and alpha. `tables/rq3_alpha_comparisons.csv` compares alpha=10 with alpha=1 for repair and breakage, correcting across methods within each mode, budget, and metric. Undefined rates are excluded from the corresponding paired test, and sample counts are recorded. The repair-time VeRe multiplier remains fixed at 1.0.
