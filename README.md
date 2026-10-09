# Weight-level fault localization for neural networks

Source code for training the benchmark models, constructing faulty models,
localizing injected faults, and evaluating localization through neural-network
repair. The supported datasets are Fashion-MNIST, CIFAR-10, and GTSRB; mutations
and repairs target the final linear layer, `fc2.weight`.

This is a local preparation draft. It contains code and configuration only.
Datasets, pretrained models, mutant checkpoints, prior experiment results, and
plotting scripts are not distributed here.

## Layout

```text
configs/             Dataset-specific experiment settings (JSON)
data/                README with dataset download and placement instructions
src/models/          Network definitions and training routines
src/data_loading/    Data loading and the model-specific preprocessing
src/mutation/        Single-fault generation and multi-fault composition
src/localization/    Localization signals, scores, and baseline adapters
src/repair/          Differential-evolution repair
src/metrics/         Localization and repair metrics
scripts/             Command-line entry points and an offline smoke test
```

## Environment

The local validation environment uses Python 3.9, PyTorch 1.13.1,
torchvision 0.14.1, NumPy 1.24.3, and Pillow 11.2.1. The corresponding package
versions are recorded in `requirements.txt`. These are the local validation
versions, not a recovered specification of the original training server.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/smoke_test.py
```

The smoke test uses synthetic tensors and a temporary directory. It checks
three model shapes, analytical gradients against autograd, the repair objective
against full-logit evaluation, and the mutation/localization/repair pipeline.
It does not download datasets or retain experiment outputs.

The experiment code chooses CUDA when available and otherwise uses CPU.
The packaged defaults use one repair worker. Set `--workers` to increase
parallelism after checking available memory. Full localization materializes
test-by-weight arrays; its memory cost is much greater than the smoke test.
Full-run runtime, peak memory, clean-environment installation, and CUDA execution
have not yet been measured or validated for this package.

## Download the datasets

The `data/` directory contains only a README with download and placement
instructions. Dataset directories are created locally when data is downloaded.
See [data/README.md](data/README.md); downloaded data is ignored by Git.

Obtain each dataset from its maintainers:

| Dataset | Download information | Default local root | Input preprocessing |
| --- | --- | --- | --- |
| Fashion-MNIST | [Official repository and downloads](https://github.com/zalandoresearch/fashion-mnist#get-the-data) | `data/fashionmnist` | 28 × 28 grayscale; tensor conversion; mean/std 0.5 |
| CIFAR-10 | [Official dataset page](https://www.cs.toronto.edu/~kriz/cifar.html) | `data/cifar10` | 32 × 32 RGB; tensor conversion; mean/std 0.5 per channel |
| GTSRB | [Benchmark maintainers' download page](https://github.com/houbensebastian/GermanTrafficSignBenchmarks#gtsrb--german-traffic-sign-recognition-benchmark) | `data/gtsrb` | Resize to 48 × 48 RGB; tensor conversion; mean/std 0.5 per channel |

The loaders use torchvision's dataset classes with `download=True`: missing
files are downloaded when a training or experiment command first loads a
dataset. Existing correctly placed data is reused. No dataset is part of the
repository. For manual installation with the pinned torchvision version, the
extracted paths are:

```text
data/fashionmnist/FashionMNIST/raw/       # The four extracted IDX files
data/cifar10/cifar-10-batches-py/         # CIFAR-10 Python batches
data/gtsrb/gtsrb/GTSRB/Training/          # Class folders containing PPM images
data/gtsrb/gtsrb/GTSRB/Final_Test/Images/
data/gtsrb/gtsrb/GT-final_test.csv
```

For GTSRB, torchvision 0.14.1 obtains `GTSRB-Training_fixed.zip`,
`GTSRB_Final_Test_Images.zip`, and `GTSRB_Final_Test_GT.zip` from the
[dataset archive](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/).
Supply all three if installing manually.

Train and test splits follow the dataset loaders. Test loading is not shuffled.
The actual model-specific preprocessing is in `src/data_loading/`.

## Configuration and paths

Use `configs/fm.json`, `configs/cifar10.json`, or `configs/gtsrb.json`.
JSON is used so that configuration loading requires no additional library.
Relative paths, including `--config`, are resolved from this artifact's root,
regardless of the current working directory. Absolute paths are also accepted.

Each entry point supports `--help`, `--dry-run`, `--data-dir`, and `--output-dir`.
Command-line options override configuration values. For example:

```bash
python scripts/run_localization.py --config configs/fm.json --dry-run
```

The generation defaults request 200 single-fault models, using 10 weight-magnitude
bins, a replacement magnitude ratio of 3, and a behavioral-change threshold of
0.001. The minimum newly broken-test count is 20 for Fashion-MNIST and 10 for
CIFAR-10/GTSRB. Multi-fault levels are 2, 3, 4, and 5 faults, requesting up to
50 models per level. Candidate shortages and rejection criteria can result in
fewer accepted models; the code reports the actual count.

These values were assembled from the inspected experiment jobs. Their final
correspondence to the submitted manuscript remains to be confirmed.

## Run the experiments

The following commands use Fashion-MNIST. Change the configuration file for
the other datasets. Runtime files go to `outputs/<dataset>/`, which is ignored
by Git. Commands refuse to overwrite an existing primary output file; choose a
new `--output-dir` for an independent run. Effective settings are saved beside
new outputs as `run_settings.json`.

### 1. Prepare a trained model

```bash
python scripts/train_model.py --config configs/fm.json
```

This writes `outputs/fm/models/fm_arachne_rq1_best_model.pth`.
The extracted training routine uses Adam, 50 epochs, a learning rate of 0.001,
and a StepLR schedule. It selects the checkpoint with the best **test-split**
accuracy, preserving the existing training procedure. The entry point additionally
sets the configured seed before training; this does not guarantee bitwise
agreement across devices or software versions.

An existing compatible checkpoint can instead be supplied using `--model-path`
in the generation command or `model_path` in the configuration. No original
experiment checkpoint is bundled. Retraining produces a new benchmark instance
and is not asserted to reproduce the historical paper numbers exactly.

### 2. Generate faulty models

```bash
python scripts/generate_mutants.py --config configs/fm.json --mode single
python scripts/generate_mutants.py --config configs/fm.json --mode multi
```

The single-fault generator selects gradient candidates, stratifies them by
weight magnitude, and tests replacement values from the same layer. The
multi-fault generator composes accepted single faults and applies the original
acceptance criteria. Outputs include `mutant_pool.json` and the corresponding
`mutant_*.pth` files under `outputs/fm/mutants_single/` or `mutants_multi/`.

### 3. Localize the faults

```bash
python scripts/run_localization.py --config configs/fm.json --mode single
python scripts/run_localization.py --config configs/fm.json --mode multi
```

Failed tests are those changing from correct to incorrect predictions. Passing
tests are those remaining correct; they are sampled at a 1:1 ratio with failed
tests, using the mutant ID as the sampling seed. Other test outcomes are excluded.
Localization ranks are 1-based and retain the original tie-ordering behavior.

The default runs both gradient and forward-impact involvement, the result-side
signals and adapted baselines, and a separate VeRe pass using the same test
selection. Results are written to
`outputs/fm/localization/<mode>/methods/faulty_ranks.json` and
`outputs/fm/localization/<mode>/vere/faulty_ranks.json`.
Use `--baseline methods` or `--baseline vere` to run only one group.

### 4. Evaluate through repair

```bash
python scripts/run_repair.py --config configs/fm.json --mode single
python scripts/run_repair.py --config configs/fm.json --mode multi
```

Repair recomputes localization on the repair half of the test split; it does not
reuse full-test localization ranks from the previous step. The other half is
held out from both localization and repair. The split seed defaults to 0.
Single-fault repair uses budgets 1, 2, 3, 5, and 8; multi-fault repair uses 2, 5,
and 10. Both use alpha values 1 and 10, a population of 100, and at most 100 DE
generations. The default examines the first 50 single-fault or 200 multi-fault
models; models with no usable repair-side failures/passes are skipped.

The output `outputs/fm/repair/<mode>/repair_results.json` includes per-model
repair and breakage counts on both splits. The terminal summary pools counts
across models; it is not a mean of per-model rates. `random` is a comparison
baseline and `truefault` is an oracle control requiring known injected faults.

For a shorter run on a pool you have already generated:

```bash
python scripts/run_repair.py --config configs/fm.json --n-mutants 2 \
    --n-fix-list 1 --alpha-list 1 --max-gen 2 --workers 1 \
    --output-dir outputs/fm/repair/quick-check
```

This reduced command checks execution only; it is not a paper experiment.

## Method provenance and release status

See [THIRD_PARTY.md](THIRD_PARTY.md) for baseline adaptations and source references.
Full upstream baseline repositories are not bundled. A project-wide license has
not yet been selected for this local draft.

Before publication, confirm the final experiment configuration, original model
provenance/training environment, dependency installation on the target platform,
and project/third-party licensing. Full dataset experiments have not been rerun
as part of this packaging work.
