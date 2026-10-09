# Methods and third-party provenance

The package contains the experiment's local implementations and adaptations.
Names such as `ArachneVar`, `DeepFaultVar`, and `GuiderVar` denote adaptations;
they must not be described as unmodified executions of the original tools.

| Method / component | Implementation in this package | Source and adaptation |
| --- | --- | --- |
| Arachne-style repair | `src/repair/differential_evolution.py` | The existing local DE implementation follows the fitness and search procedure described in [Arachne](https://github.com/coinse/arachne), including the referenced `search/de_vk.py` and `searcher_vk.py`. Fitness evaluations specialize to updates of `fc2.weight`. This package does not include the upstream TensorFlow/Keras tool. |
| ArachneVar | `src/localization/signals.py`, `runner.py` | Gradient involvement with the binary result-side variant (`G_QB_cos_ochiai`). This is the experiment's weight-level comparison variant, not Arachne's original multi-objective localization procedure. |
| DeepFault / DeepFaultVar | `src/localization/signals.py`, `runner.py` | Ochiai-based neuron coverage and a weight-level top-impact coverage adaptation. Neuron scores are expanded over outgoing weights for comparison at `fc2`. Reference: [DeepFault paper](https://arxiv.org/abs/1902.05974). |
| Guider / GuiderVar | `src/localization/signals.py`, `runner.py` | Confidence-weighted Ochiai and its weight-level coverage adaptation. Reference: [Weighted Suspiciousness and Balanced Aggregation to Boost Spectrum-based Fault Localization of Deep Learning Models](https://doi.org/10.1145/3716849). |
| VeRe | `src/localization/vere.py`, `vere_runner.py` | Repair-significance localization specialized to a linear final-layer suffix, using analytical endpoint effects. It is not the full verification/synthesis repair pipeline. Reference: [VeRe repository](https://github.com/nninjn/VeRe) and [paper](https://doi.org/10.1145/3597503.3623332). |
| Multi-fault benchmark construction / weight coverage | `src/mutation/multi.py`, `src/localization/signals.py` | The existing implementation uses cumulative-fault acceptance and top-impact coverage inspired by [SpectAcle](https://doi.org/10.1145/3705307), adapted to these classifiers and explicit fault counts. |

VeRe localization evaluates interval multipliers 1, 1.5, and 2 and writes all
three rank sets. The repair comparison uses multiplier 1.0, following the
existing repair code. These are different experimental settings and should be
identified when reporting comparisons.

Dataset links and preprocessing are documented in [README.md](README.md).
Dataset files and complete upstream baseline repositories are not redistributed
in this package. Runtime dependencies are installed separately.

## Licensing status

This attribution file is not a project license. The main license and any required
upstream copyright/license notices still need to be settled before distributing
the local draft. These references do not imply a blanket license covering all
listed sources.
