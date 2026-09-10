# MAAD-Bench

MAAD-Bench is a model-agnostic benchmark for audio anomaly detection. It places
seven public audio sub-datasets under one 74-feature tabular schema, 25 PyOD
detectors, and a group-rotated evaluation protocol. The revision uses every native
group once as the held-out test set, for 59 leave-one-group-out folds, and reports
variation across folds rather than variation across repeated seeds of one split.

## Benchmark scope

| Anomaly regime | Sub-datasets | Evaluation groups |
|---|---|---|
| Designated class | ESC-50, UrbanSound8K, RAVDESS, NSynth valid split | 5 classes, 10 official folds, 24 actors, 5 instrument buckets |
| Organic label | MIMII 6 dB pump subset, PhysioNet 2016, ICBHI 2017 | 4 machine IDs, 6 collection subsets, 5 patient buckets |

Each clip becomes a 74-dimensional vector: 20 MFCCs, 12 chroma bins, and five
spectral descriptors, each summarized by its mean and standard deviation. The
primary setting trains on normal clips only. The secondary setting adds 10%
training contamination. Both use five seeds, while the reported uncertainty is
the standard deviation across held-out groups.

## Installation

Python 3.12 was used for the reported results.

```text
pip install -r requirements.txt
```

`requirements.txt` records the exact DGX Spark environment. Its Torch build was a
CUDA 12.8 ARM64 wheel; install the Torch build appropriate for your platform when
rerunning neural detectors. PANNs and AST packages are needed only to regenerate
embeddings from audio. The 14 cached embedding matrices are included.

## Released artifacts

The raw corpora are not redistributed. Download them from their original sources
and place them below `data/` if you need to extract features again.

| Path | Contents |
|---|---|
| `results/<dataset>/features.parquet` | Released handcrafted feature matrix |
| `results/<dataset>/metadata.parquet` | Row-aligned labels, native group, category, and relative source path |
| `results/<dataset>/splits.parquet` | Explicit train/test fold assignment |
| `results/<dataset>/rot_runs.parquet` | Per-run metrics for the revised group rotation |
| `results/<dataset>/rot_metrics.csv` | Fold-aggregated revised main results |
| `results/<dataset>/rot_clip_scores.parquet` | One score per held-out clip, detector, and seed |
| `results/<dataset>/rot_{panns,ast}_*.{csv,parquet}` | Fold-local pretrained-representation runs and summaries |
| `results/<dataset>/rot_temporal_*.{csv,parquet}` | Frame-level temporal runs and summaries |
| `results/analysis_*` | Reliability, transfer, group-shift, and ceiling analyses |
| `embeddings/<dataset>_{panns,ast}.parquet` | Cached PANNs and AST embeddings |

The files `metrics.csv`, `scores.parquet`, and `config.json` retain the submitted
fixed-fold run for audit history. Results used in the revision have the `rot_`
prefix or an `analysis_` prefix.

The release contains 4,493,097 per-clip score rows. RAVDESS is 28 rows short of
the nominal total because one CBLOF fit failed for the fold holding actor 10 at
seed 3; the error is recorded in `rot_runs.parquet`, and the other 25 by 5 passes
cover every test clip exactly once.

## Verify the packaged result tree

Run this first. It checks all seven sub-datasets, fold coverage, detector and seed
counts, the recorded failure, and score-path resolution against metadata.

```text
python check_clip_scores.py
```

Useful read-only analysis commands include:

```text
python analysis.py --repr handcrafted --setting primary
python stats_cd.py --repr handcrafted --setting primary
python origin_transfer.py --repr handcrafted --setting primary
python gen_ae_anchor.py
python gen_table8.py
python gen_table11.py
python w2_transfer.py
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

## Data sources

| Sub-dataset | Source |
|---|---|
| ESC-50 | <https://github.com/karolpiczak/ESC-50> |
| UrbanSound8K | <https://urbansounddataset.weebly.com/urbansound8k.html> |
| RAVDESS | <https://zenodo.org/record/1188976> |
| NSynth | <https://magenta.tensorflow.org/datasets/nsynth> |
| MIMII | <https://zenodo.org/record/3384388> |
| PhysioNet 2016 | <https://physionet.org/content/challenge-2016/> |
| ICBHI 2017 | <https://bhichallenge.med.auth.gr/> |

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
