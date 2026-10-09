# Dataset download instructions

This directory contains only this guide. Download the datasets separately;
their local directories are created during downloading or manual installation.
Downloaded data is ignored by Git.

## Download sources

| Dataset | Download page | Local root |
| --- | --- | --- |
| Fashion-MNIST | [Official repository and downloads](https://github.com/zalandoresearch/fashion-mnist#get-the-data) | `data/fashionmnist/` |
| CIFAR-10 | [Official dataset page](https://www.cs.toronto.edu/~kriz/cifar.html) | `data/cifar10/` |
| GTSRB | [Benchmark maintainers' page](https://github.com/houbensebastian/GermanTrafficSignBenchmarks#gtsrb--german-traffic-sign-recognition-benchmark) | `data/gtsrb/` |

The data loaders use torchvision with `download=True`. Running a training or
experiment command downloads any missing files; correctly placed existing data
is reused. Downloading has not been performed as part of preparing this repository.

For manual installation with the pinned torchvision version, use these extracted
paths relative to the repository root:

```text
data/fashionmnist/FashionMNIST/raw/
    train-images-idx3-ubyte
    train-labels-idx1-ubyte
    t10k-images-idx3-ubyte
    t10k-labels-idx1-ubyte

data/cifar10/cifar-10-batches-py/
    data_batch_1 ... data_batch_5
    test_batch
    batches.meta

data/gtsrb/gtsrb/GTSRB/Training/
data/gtsrb/gtsrb/GTSRB/Final_Test/Images/
data/gtsrb/gtsrb/GT-final_test.csv
```

GTSRB requires the training images, final test images, and final test ground
truth. The pinned torchvision loader uses `GTSRB-Training_fixed.zip`,
`GTSRB_Final_Test_Images.zip`, and `GTSRB_Final_Test_GT.zip` from the
[dataset archive](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/).

To use data stored elsewhere, pass `--data-dir /path/to/dataset-root` to the
entry point or update `data_dir` in the corresponding configuration. See the
[main README](../README.md) for preprocessing and execution instructions.
