# Preparing the datasets

NeuroNTP reads data through the `EpilepsiaCoreDataset` / `EpilepsiaDatasetView` abstraction. Before training, each dataset goes through three steps:

1. **Obtain** the raw dataset (EDF files and annotations) from its original source, and **verify** that it is complete.
2. **Convert** it to HDF5: one compressed file per recording or session, plus CSV index tables. This is the only step that touches the signal data.
3. **Create** the core dataset and dataset views used for pretraining, fine-tuning and evaluation. These files contain metadata only (patients, sessions, annotations, sample and chunk indices) and point to the HDF5 store from step 2.

Steps 1 and 2 need to be done **once per dataset**. Step 3 is fast by comparison and can be repeated later, for example to create alternative views with a different configuration (`--view_cfg <config>.yaml`).
Because of the storage the raw data and the HDF5 store require, downloading and converting the datasets (steps 1 and 2) is not part of the pipeline configurations; each dataset is downloaded and converted once, as described below.

The download, conversion and view creation scripts are located in the `scripts/` folder.

| Dataset | Used for | Version | Conversion | Core dataset and views | HDF5 store size* |
|---|---|---|---|---|------------------|
| Siena Scalp EEG Database | fine-tuning, evaluation | 1.0.0 | `siena2hdf.py` | `create_siena_ds.py` | ~13 GB           |
| CHB-MIT Scalp EEG Database | fine-tuning, evaluation | 1.0.0 | `chb2hdf.py` | `create_chb_ds.py` | ~39 GB           |
| TUH EEG Seizure Corpus (TUSZ) | pretraining only | **v2.0.3** | `tusz2hdf.py` | `create_tusz_ds.py` | ~129 GB          |

\* With default settings, measured as apparent size (`du -sh --apparent-size`); the space used on disk can be larger, depending on the file system. The raw downloads need additional space on top of this. You can delete them once conversion has finished, although keeping them lets you re-run the conversion with different settings.

## Before you start

**Environment.** The preprocessing scripts import the dataset classes from the `neuro_ntp` package, so the NeuroNTP environment must be installed and activated. From the repository root:

```bash
conda env create -f environment.yml
conda activate neurontp
```

This installs `neuro_ntp` in editable mode with all its dependencies, including those of the preprocessing scripts. The download script (`download_data.py`) uses the Python standard library only; for TUSZ, `rsync` must be installed.

**Disk space.** Make sure the target drive has enough room for the HDF5 stores (see the table) plus the raw data. A fast local drive (SSD) is recommended, since training reads the HDF5 files directly.

**Directory layout.** The core datasets and views store the **paths** of the HDF5 files. They must be created on the system where the HDF5 store resides, and the store should not be moved afterwards. Choose a stable location first. We suggest:

```
datasets/
├── raw/                  # original downloads (step 1)
│   ├── siena/1.0.0/
│   ├── chbmit/1.0.0/
│   └── tusz/v2.0.3/
├── hdf5/                 # HDF5 stores (step 2)
│   ├── siena/
│   ├── chbmit/
│   └── tusz/
└── views/                # core datasets and views (step 3), referenced in training configs
    ├── siena/
    ├── chbmit/
    └── tusz/
```

The commands below follow this layout. Adjust the paths if you use a different one, and reference the files in `datasets/views/` from your training and evaluation configurations.

## Downloading and verifying the raw data

Incomplete downloads are the most common cause of results that do not reproduce. Missing recordings silently shrink the training set, and you only notice later, as deviating results. `scripts/download_data.py` downloads the datasets and checks **every file against a SHA-256 checksum**:

```bash
python scripts/download_data.py --datasets siena chbmit tusz --out_root datasets/raw     # or: --datasets all
```

- **Checksums.** Siena and CHB-MIT are checked against PhysioNet's `SHA256SUMS.txt`. TUSZ has no published checksums, so it is checked against `res/manifests/tusz_v2.0.3.sha256`, a manifest of the copy used in the paper (see the TUSZ section).
- **Re-running.** The script is safe to re-run. Verified files are skipped, interrupted downloads resume, and corrupted files are fetched again. A file is only moved into place after its checksum has been verified. Re-run the script until every dataset reports `OK`.
- **Existing downloads.** `--verify_only` checks an existing download without fetching anything. If you downloaded the data some other way (e.g. with `wget`), run this before converting.
- **Failures.** Problem files are listed in `datasets/raw/<dataset>/<version>/VERIFY_FAILED.txt`. The exit status is non-zero until all requested datasets are complete.
- **Further options.** `--jobs` sets the number of parallel downloads and checksum workers (default 8), and `--retries` the number of attempts per file (default 5).

**Do not start the conversion (step 2) before the dataset reports `OK`.**

## Siena Scalp EEG Database

### 1. Obtain the data

The database is available from [PhysioNet](https://physionet.org/content/siena-scalp-eeg/1.0.0/) under the CC BY 4.0 license:

```bash
python scripts/download_data.py --datasets siena --out_root datasets/raw
```

<details>
<summary>Alternative: plain <code>wget</code> (verify afterwards)</summary>

```bash
wget -r -N -c -np -nH --cut-dirs=2 -P datasets/raw/siena https://physionet.org/files/siena-scalp-eeg/1.0.0/
python scripts/download_data.py --datasets siena --out_root datasets/raw --verify_only
```
`wget -r` can skip files silently, for example on timeouts. Always run the verification.
</details>

The expected layout is `<siena_root>/PNxx/PNxx-N.edf` with one `Seizures-list-PNxx.txt` per patient, where `<siena_root>` is `datasets/raw/siena/1.0.0`.

### 2. Convert to HDF5

```bash
python scripts/siena2hdf.py datasets/raw/siena/1.0.0 --out_hdf_root "$(pwd)/datasets/hdf5/siena" --n_jobs 8
```
- Channel names are normalised, and only supported channels are kept: 31 EEG electrodes, plus SpO2, heart rate and plethysmogram.
- Signals are resampled to 256 Hz and stored as float32 without further filtering.
- Seizure intervals are taken from the seizure lists and converted to seconds relative to the start of each recording.

### 3. Create the core dataset and views

```bash
python scripts/create_siena_ds.py --hdf_root datasets/hdf5/siena --out_dir datasets/views/siena
```

| File | Purpose |
|---|---|
| `siena_core.h5` | core dataset |
| `siena_selfsup_dv.h5` | default self-supervised view with patient-level split (test: PN14, PN16, PN17; val: PN11, PN12, PN13) |
| `siena_selfsup_dv_bal.h5` | balanced view for fine-tuning: ictal samples plus an equal number of non-ictal samples per patient; test patients excluded |
| `siena_selfsup_16_17_dv.h5` | test view following the reference protocol (test: PN16, PN17) |

**Reproducing the fine-tuning selection.** To use the exact selection from the paper, pass the released template with the `--bal_template` command line argument (optional; recommended for reproduction).
```bash
python scripts/create_siena_ds.py --hdf_root datasets/hdf5/siena --out_dir datasets/views/siena \
    --bal_template res/view_template/siena_selfsup_dv_bal_template.h5
```
The template is checked against the sampling rates and signal lengths of your HDF5 files. This only works if step 2 was run with the default settings.


## CHB-MIT Scalp EEG Database

### 1. Obtain the data

The database is available from [PhysioNet](https://physionet.org/content/chbmit/1.0.0/) under the Open Data Commons Attribution License v1.0:

```bash
python scripts/download_data.py --datasets chbmit --out_root datasets/raw
```

<details>
<summary>Alternative: plain <code>wget</code> (verify afterwards)</summary>

```bash
wget -r -N -c -np -nH --cut-dirs=2 -P datasets/raw/chbmit https://physionet.org/files/chbmit/1.0.0/
python scripts/download_data.py --datasets chbmit --out_root datasets/raw --verify_only
```
`wget -r` can skip files silently, for example on timeouts. Always run the verification.
</details>

The expected layout is `<chb_root>/chbXX/chbXX_YY.edf` with one `chbXX-summary.txt` per patient, where `<chb_root>` is `datasets/raw/chbmit/1.0.0`.

### 2. Convert to HDF5

```bash
python scripts/chb2hdf.py datasets/raw/chbmit/1.0.0 --out_hdf_root "$(pwd)/datasets/hdf5/chbmit" --n_jobs 8
```

- The conversion first checks that all 24 patients are present with their `chbXX-summary.txt` file, and that every recording listed in the summary files is present as EDF. It stops with a list of the missing files otherwise. `--allow_missing` converts an incomplete copy anyway; this is not recommended, since the result will not match the paper.
- Signals are kept at the native 256 Hz without filtering. Use `--sampling_freq` to resample.
- Seizure intervals are taken from the per-patient summary files.
- chb01 and chb21 were recorded from the same subject 1.5 years apart. Take this into account if you define your own patient-level splits.

### 3. Create the core dataset and views

```bash
python scripts/create_chb_ds.py --hdf_root datasets/hdf5/chbmit --out_dir datasets/views/chbmit [--proper]
```

| File | Purpose |
|---|---|
| `chbmit_core_ds.h5` | core dataset |
| `chbmit_selfsup_10s_biot.h5` | default view, compatible with the BIOT/CBraMod reference protocol (10 s windows; train: chb01–chb20, val: chb21–chb22, test: chb23–chb24) |
| `chbmit_selfsup_10s_proper.h5` | as above, but with corrected ictal labelling (only with `--proper`) |

The default view reproduces the reference protocol, including its labelling, augmentation and file filtering. The view configuration lets you switch each of these off individually (`biot_compat`, `augment_ictal`, `apply_biot_process1_file_filter`). The differences are discussed in the NeuroNTP paper.

**Reproducing the sample selection.** The default view contains a random selection of samples, including the augmented ictal samples of the reference protocol. To use the exact selection from the paper, pass the released template with the `--template` command line argument (optional; recommended for reproduction) instead of building a new default view:

```bash
python scripts/create_chb_ds.py --hdf_root datasets/hdf5/chbmit --out_dir datasets/views/chbmit \
    --template res/view_template/chbmit_selfsup_10s_biot_template.h5
```

The template is checked against your HDF5 store: every referenced file must exist and match the core dataset in sampling rate, signal length and sample ranges. If a recording is missing or was converted with different settings, the run stops with a list of the affected files instead of silently producing a smaller dataset. This only works if step 2 was run with the default settings. The proper view (`--proper`) is always built anew.

## TUH EEG Seizure Corpus (TUSZ)

### 1. Obtain the data

TUSZ is distributed by the Neural Engineering Data Consortium (NEDC) under the TUH data use agreement. [Register for access](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/); the password for the download is sent to you by e-mail. Then download **v2.0.3** with rsync and verify it:

```bash
python scripts/download_data.py --datasets tusz --out_root datasets/raw
```

This downloads into `datasets/raw/tusz/v2.0.3/` and asks for the NEDC password.

- **Version.** Use v2.0.3. Later releases changed the data substantially: v2.0.5, for example, re-annotated the corpus and limited recordings to 1800 s. Results are only comparable with v2.0.3, which has 7,364 EDF files (train 4,667, dev 1,832, eval 865).
- **Rsync address.** By default, the script downloads from `nedc-eeg@www.isip.piconepress.com:data/eeg/tuh_eeg_seizure/v2.0.3/`, following NEDC's [download instructions](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml). NEDC changes this address occasionally. If the download fails, list the available corpora with `rsync nedc-eeg@www.isip.piconepress.com:data/eeg/` and pass the correct source with `--tusz_src`, or update `TUSZ_RSYNC_SRC` at the top of `scripts/download_data.py`:
  ```bash
  python scripts/download_data.py --datasets tusz --out_root datasets/raw \
      --tusz_src nedc-eeg@www.isip.piconepress.com:data/eeg/<corpus_dir>/v2.0.3/
  ```
- **Files updated by NEDC.** NEDC has updated v2.0.3 in place (header, corrupted-file and annotation fixes; last change 2025-06-18), so a current download can differ from our copy in a few files. Such files are reported as a warning and listed in `datasets/raw/tusz/v2.0.3/CHANGED_FILES.txt`, and small deviations from the reported results are possible. Only missing files count as an error. Use `--tusz_strict` to treat changed files as errors too.
- **Existing copy.** If you already have TUSZ v2.0.3, place it so that `datasets/raw/tusz/v2.0.3/edf/` exists, and run the script with `--verify_only`.

The expected layout is `edf/{train,dev,eval}/<patient>/<session>/<montage>/<patient>_<session>_tNNN.{edf,csv,csv_bi}`.

### 2. Convert to HDF5

```bash
python scripts/tusz2hdf.py --tusz_edf_root datasets/raw/tusz/v2.0.3/edf --out_hdf_root "$(pwd)/datasets/hdf5/tusz" --n_jobs 16
```

- The EDF segments of each session are resampled to 256 Hz, stitched together, and filtered with a 0.5–100 Hz band-pass and a 60 Hz notch filter. Disable the filters with `--bandpass none` or `--notch none`.
- Sessions shorter than 300 s are skipped and listed in `skipped.csv`.
- Use `--require_same_channels` to check channel consistency across the segments of a session.

### 3. Create the core dataset and view

```bash
python scripts/create_tusz_ds.py --hdf_root datasets/hdf5/tusz --out_dir datasets/views/tusz
```

| File | Purpose |
|---|---|
| `TUSZ_coreds.h5` | core dataset |
| `dv_selfsup_10s.h5` | pretraining view (10 s chunks) |

TUSZ is used for self-supervised pretraining only. During training, samples are drawn randomly from the view, so the view defines a pool of samples rather than a fixed sequence.

## General notes

- **Completeness.** Run the conversion only on a verified download (`download_data.py` reports `OK`). The released view templates check the HDF5 store again in step 3, but a dataset view built without a template cannot detect missing recordings.
- **Re-running.** The `create_*_ds.py` scripts reuse existing core dataset and default view files unless `--overwrite` is given. Derived views (the Siena balanced and test views, and the CHB-MIT proper view) are always rebuilt; views localized from a template (`--bal_template`, `--template`) are always re-created from the template. After fixing an incomplete download, re-run the conversion and then step 3 with `--overwrite`, since the core dataset is built from the HDF5 store.
- **Absolute paths.** The HDF5 file paths are recorded in `sessions.csv` and `annotations.csv` and carried into the core datasets and views, so pass an absolute path to `--out_hdf_root` in step 2 (e.g. `--out_hdf_root "$(pwd)/datasets/hdf5/siena"`), as in the commands above.
- **Moving data.** If you move the HDF5 store, re-create the core datasets and views (step 3) on the new location.
- **Licenses.** None of the datasets are included in this repository. Please follow the license terms of each source.

## Preparing your own dataset

To use your own data with NeuroNTP, follow the same three steps:

1. **Convert** your recordings to one HDF5 file per recording or session. Each file should contain a `signal` array (`n_channels × n_samples`) and the attributes `sfreq`, `signal_length`, `n_channels`, `ch_names`, `patient_id` and the record identifier. Annotations go in `annotations.{onset,duration,description}`, with times in seconds from the start of the recording.
2. **Write the index tables** `sessions.csv` (one row per file) and `annotations.csv` (one row per annotation) in the HDF5 root.
3. **Create** a core dataset and view by subclassing `EpilepsiaCoreDataset` and `EpilepsiaDatasetView`. Use `siena2hdf.py` and `create_siena_ds.py` as a template, since they are the simplest of the three pipelines.

Normalise channel names to the common scheme (`"eeg fp1"`, `"spo2"`, `"hr"`, ...) so the tokenizer can match channels across datasets.