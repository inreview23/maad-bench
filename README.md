# MAAD-Bench

MAAD-Bench is a model-agnostic benchmark for audio anomaly detection. It places
14 public audio sub-datasets under one 74-feature tabular schema, 25 PyOD
detectors, and a group-rotated evaluation protocol. Each evaluation group is held out once, for 174 leave-one-group-out folds. Groups follow the native split, actor, machine ID, or source database where available; NSynth uses instrument-identifier buckets, and ICBHI and CirCor use patient-identifier buckets. The reported spread is the variation across folds rather than across repeated seeds of one split.

## Benchmark scope

| Anomaly regime | Sub-datasets | Evaluation groups (folds) |
|---|---|---|
| Designated class | ESC-50, UrbanSound8K, RAVDESS, NSynth valid split, CREMA-D | 5 official folds, 10 official folds, 24 actors, 5 instrument buckets, 91 actors |
| Organic label | MIMII 6 dB pump, fan, slider and valve; DCASE 2020 Task 2 ToyCar and ToyConveyor; PhysioNet 2016; ICBHI 2017; CirCor DigiScope (recording-level labels) | 4 machine IDs each; 4 and 3 machine IDs; 6 source databases; 5 patient buckets; 5 patient buckets |

Each clip becomes a 74-dimensional vector: 20 MFCCs, 12 chroma bins, and five
spectral descriptors, each summarized by its mean and standard deviation. The
primary setting trains on normal clips only. The secondary setting adds anomalies
until they make up 10% of the training clips. Both use five seeds.

The seven sub-datasets added in this revision are CREMA-D, MIMII fan, slider and
valve, ToyCar, ToyConveyor, and CirCor (`circor_rec`). CirCor labels a recording
anomalous only when the patient has a murmur and the recording was taken at a
location listed in that patient's murmur locations; recordings of murmur patients
at other locations and recordings of patients with an unknown murmur status are
excluded.

## Installation

Python 3.12 was used for the reported results.

```text
pip install -r requirements.txt
```

`requirements.txt` records the exact DGX Spark environment. Its Torch build was a
CUDA 12.8 ARM64 wheel; install the Torch build appropriate for your platform when
rerunning neural detectors. PANNs and AST packages are needed only to regenerate
embeddings from audio. The cached embedding matrices of the original seven
sub-datasets are included; those of the seven added sub-datasets are regenerated
from audio with `e6_repr.py`.

## Released artifacts

The raw corpora are not redistributed. Download them from their original sources
and place them below `data/` if you need to extract features again.

| Path | Contents |
|---|---|
| `results/<dataset>/features.parquet` | Released handcrafted feature matrix |
| `results/<dataset>/metadata.parquet` | Row-aligned labels, native group, category, and relative source path |
| `results/<dataset>/splits.parquet` | Explicit train/test fold assignment |
| `results/<dataset>/rot_runs.parquet` | Per-run metrics for the group rotation |
| `results/<dataset>/rot_metrics.csv` | Fold-aggregated main results |
| `results/<dataset>/rot_clip_scores.parquet` | One score per held-out clip, detector, and seed (primary setting) |
| `results/<dataset>/rot_{panns,ast}_*.{csv,parquet}` | Fold-local pretrained-representation runs and summaries |
| `results/<dataset>/ae_rot_*.{csv,parquet}` | Log-mel autoencoder anchor runs and summary |
| `results/analysis_r3/` | Scripts and outputs behind the tables of the 14-sub-dataset manuscript |
| `embeddings/<dataset>_{panns,ast}.parquet` | Cached PANNs and AST embeddings (original seven) |

The release contains 8,654,572 per-clip score rows over the 14 sub-datasets. Three
of the 43,500 detector fits failed, each a CBLOF fit (RAVDESS, CREMA-D, and MIMII
valve); each failure is recorded in `rot_runs.parquet`, and the fold mean uses the
other four seeds.

## Verify the packaged result tree

`check_clip_scores.py` checks the original seven sub-datasets: fold coverage,
detector and seed counts, the recorded failure, and score-path resolution against
metadata.

```text
python check_clip_scores.py
```

Useful read-only analysis commands include:

```text
python analysis.py --repr handcrafted --setting primary
python stats_cd.py --repr handcrafted --setting primary
python origin_transfer.py --repr handcrafted --setting primary
python gen_ae_anchor.py
```

To refit the main detector roster from the released features:

```text
python rotate.py --dataset all --jobs 1
```

To refit the detector roster on the cached PANNs and AST embeddings:

```text
python rotate_repr.py --dataset all --backbone panns --jobs 1
python rotate_repr.py --dataset all --backbone ast --jobs 1
```

The temporal and log-mel autoencoder arms require per-frame caches that are not
included because they are several gigabytes. Their per-run outputs are included,
so the corresponding tables can be recomputed but those two arms cannot be
refitted from this release without the raw audio and regenerated caches.

## Scripts Added in This Revision

| Script | Purpose |
|---|---|
| `e4_corpora.py`, `extract_e4_features.py`, `summarize_e4.py` | CREMA-D and CirCor loaders, features, rotation |
| `e5_circor_rec.py` | CirCor recording-level labels and campaign analysis |
| `e1_mimii_types.py`, `e1_runner.py` | MIMII fan, slider and valve |
| `e2_dcase.py`, `tabulate_e2.py`, `generate_e2b_artifacts.py`, `download_and_extract.py` | DCASE 2020 Task 2 development split and the ToyCar and ToyConveyor sub-datasets |
| `e3_hpsens.py`, `e3_analysis.py` | Hyperparameter grid fixed in advance |
| `e6_repr.py`, `e6_runner.py` | PANNs and AST embeddings with fold-local PCA (through the existing `rotate_repr.py`) |
| `e7_ae.py` | Log-mel autoencoder anchor for CREMA-D and CirCor |
| `download_circor.py` | CirCor download from the PhysioNet open-data bucket |
| `results/analysis_r3/R1F/r1_all.py` | Per-task table, loudness baseline, max-of-25 null, significance tests |
| `results/analysis_r3/R2F/r2_all.py` | Ordering reproducibility, cross-task agreement, ranks, effective detectors |
| `results/analysis_r3/make_tables.py` | Generates the manuscript tables and figures from the files above |
| `results/analysis_r3/P1/inductive_subset.py` | Main results over the 20 detectors that score each clip against the training model alone |

## Data sources

| Sub-dataset | Source |
|---|---|
| ESC-50 | <https://github.com/karolpiczak/ESC-50> |
| UrbanSound8K | <https://urbansounddataset.weebly.com/urbansound8k.html> |
| RAVDESS | <https://zenodo.org/record/1188976> |
| NSynth | <https://magenta.tensorflow.org/datasets/nsynth> |
| CREMA-D | <https://github.com/CheyneyComputerScience/CREMA-D> |
| MIMII (pump, fan, slider, valve) | <https://zenodo.org/record/3384388> |
| DCASE 2020 Task 2 development set (ToyCar, ToyConveyor) | <https://zenodo.org/records/3678171> |
| PhysioNet 2016 | <https://physionet.org/content/challenge-2016/> |
| ICBHI 2017 | <https://bhichallenge.med.auth.gr/> |
| CirCor DigiScope 1.0.3 | <https://physionet.org/content/circor-heart-sound/1.0.3/> |

## Citation

```bibtex
@unpublished{maadbench2026,
  title  = {MAAD-Bench: A Model-Agnostic Benchmark for Audio Anomaly Detection},
  author = {Qian Hao and Liang Zhao},
  note   = {Manuscript submitted for publication},
  year   = {2026}
}
```

## License

Released under the MIT License. See `LICENSE`.
