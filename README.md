# Weight level fault localization for neural networks

**Supplementary proof:** [Pass/fail score ordering of the five outcome measures — Property 1](PROPERTY_PROOF.md)

This repository provides the implementation and experimental code for weight-level fault localization in neural networks. The experiments use Fashion-MNIST, CIFAR-10, and GTSRB, and cover faulty model construction, fault localization, and neural-network repair.

## System Requirements

- Python: 3.9
- PyTorch: 1.13.1
- torchvision: 0.14.1
- NumPy: 1.24.3
- Pillow: 11.2.1

The scripts use CUDA when available and CPU otherwise. Dependency versions are specified in `requirements.txt`.

## Installation

From the repository root, create a virtual environment and install the dependencies:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Preparation

### Datasets

The experiments use the following datasets and configuration files:

| Dataset | Download | Local directory | Configuration |
| --- | --- | --- | --- |
| Fashion-MNIST | [Official repository](https://github.com/zalandoresearch/fashion-mnist#get-the-data) | `data/fashionmnist` | `configs/fm.json` |
| CIFAR-10 | [Official dataset page](https://www.cs.toronto.edu/~kriz/cifar.html) | `data/cifar10` | `configs/cifar10.json` |
| GTSRB | [Download information](https://github.com/houbensebastian/GermanTrafficSignBenchmarks#gtsrb--german-traffic-sign-recognition-benchmark) | `data/gtsrb` | `configs/gtsrb.json` |

Datasets are downloaded automatically when first accessed. See [data/README.md](data/README.md) for manual installation.

The commands below use Fashion-MNIST. Replace `configs/fm.json` with `configs/cifar10.json` or `configs/gtsrb.json` to run the other datasets.

### Model Training

Train the benchmark model:

```bash
python scripts/train_model.py --config configs/fm.json
```

The trained model is saved under `outputs/fm/models/`. Pretrained checkpoints are not included.

### Construction of Benchmarks

Generate single-fault models, then construct multiple-fault models from the accepted single faults:

```bash
python scripts/generate_mutants.py --config configs/fm.json --mode single
python scripts/generate_mutants.py --config configs/fm.json --mode multi
```

The resulting checkpoints and `mutant_pool.json` files are stored under:

```text
outputs/fm/mutants_single/
outputs/fm/mutants_multi/
```

## Usage

Run the commands below from the repository root. Experiment configurations are stored in `configs/`. Use `--help` to view the available options.

Each experiment saves its results in the specified output directory. Use a new output directory when repeating an experiment.

## Reproduction of Experimental Results

### RQ1: Which Combination of Our Approach Performs Best?

RQ1 compares ten configurations combining two involvement measures with five result-side measures. Run localization on the single-fault and multiple-fault pools:

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

The configuration selected in RQ1, `G_Cos`, is used as our method in RQ2 and RQ3.

### RQ2: How Does Our Method Compare with Existing Methods?

RQ2 compares our method with the existing methods and their weight-level adaptations. Run the localization methods and the VeRe baseline on both pools:

```bash
python scripts/run_localization.py --config configs/fm.json --mode single \
    --baseline all --execution both --output-dir outputs/fm/rq2/single
python scripts/run_localization.py --config configs/fm.json --mode multi \
    --baseline all --execution both --output-dir outputs/fm/rq2/multi
```

Results are saved under `outputs/fm/rq2/<mode>/methods/` and `outputs/fm/rq2/<mode>/vere/`.

The comparison includes our method, ArachneVar, DeepFaultVar, GuiderVar, DeepFault, Guider, and VeRe. See [THIRD_PARTY.md](THIRD_PARTY.md) for implementation details.

### RQ3: Does More Accurate Localisation Lead to Better Repair?

RQ3 compares the repair outcomes obtained by selecting weights with different localization methods. All methods use a shared differential-evolution repair procedure:

```bash
python scripts/run_repair.py --config configs/fm.json --mode single \
    --output-dir outputs/fm/rq3/single
python scripts/run_repair.py --config configs/fm.json --mode multi \
    --output-dir outputs/fm/rq3/multi
```

The comparison includes the localization methods from RQ2, random weight selection, and an oracle that selects the injected faulty weights. Repair is evaluated on a held-out split.

Results are saved to:

```text
outputs/fm/rq3/single/repair_results.json
outputs/fm/rq3/multi/repair_results.json
```
