# Preparing the datasets

NeuroNTP reads data through the `EpilepsiaCoreDataset` / `EpilepsiaDatasetView` abstraction. Before training, each dataset goes through three steps:

1. **Obtain** the raw dataset (EDF files and annotations) from its original source.
2. **Convert** it to HDF5: one compressed file per recording or session, plus CSV index tables. This is the only step that touches the signal data.
3. **Create** the core dataset and dataset views used for pretraining, fine-tuning and evaluation. These files contain metadata only (patients, sessions, annotations, sample and chunk indices) and point to the HDF5 store from step 2.

Steps 1 and 2 need to be done **once per dataset**. Step 3 is fast by comparison and can be repeated later, for example to create alternative views with a different configuration (`--view_cfg <config>.yaml`).

| Dataset | Used for | Conversion | Core dataset and views | HDF5 store size* |
|---|---|---|---|---|
| Siena Scalp EEG Database | fine-tuning, evaluation | `siena2hdf.py` | `create_siena_ds.py` | ~17 GB |
| CHB-MIT Scalp EEG Database | fine-tuning, evaluation | `chb2hdf.py` | `create_chb_ds.py` | ~160 GB |
| TUH EEG Seizure Corpus (TUSZ) | pretraining only | `tusz2hdf.py` | `create_tusz_ds.py` | ~173 GB |

\* With default settings. The raw downloads need additional space on top of this. You can delete them once conversion has finished, although keeping them lets you re-run the conversion with different settings.

## Before you start

**Environment.** The preprocessing scripts import the dataset classes from the `neuro_ntp` package, so the NeuroNTP environment must be installed and activated. From the repository root:

```bash
conda env create -f environment.yml
conda activate neurontp
```

This installs `neuro_ntp` in editable mode with all extras. The preprocessing steps themselves only need the core package plus the `torch` and `eeg` extras (`pip install -e ".[torch,eeg]"`).

**Disk space.** Make sure the target drive has enough room for the HDF5 stores (see the table) plus the raw data. A fast local drive (SSD) is recommended, since training reads the HDF5 files directly.

**Directory layout.** The core datasets and views store the **paths** of the HDF5 files. They must be created on the system where the HDF5 store resides, and the store should not be moved afterwards. Choose a stable location first. We suggest:

```
datasets/
├── raw/                  # original downloads (step 1)
│   ├── siena/
│   ├── chbmit/
│   └── tusz/
├── hdf/                  # HDF5 stores (step 2)
│   ├── siena/
│   ├── chbmit/
│   └── tusz/
└── views/                # core datasets and views (step 3), referenced in training configs
    ├── siena/
    ├── chbmit/
    └── tusz/
```

The commands below follow this layout. Adjust the paths if you use a different one, and reference the files in `datasets/views/` from your training and evaluation configurations.

## Siena Scalp EEG Database

### 1. Obtain the data

The database is available from [PhysioNet](https://physionet.org/content/siena-scalp-eeg/1.0.0/) under the CC BY 4.0 license:

```bash
wget -r -N -c -np -nH --cut-dirs=2 -P datasets/raw/siena https://physionet.org/files/siena-scalp-eeg/1.0.0/
```

The expected layout is `<siena_root>/PNxx/PNxx-N.edf` with one `Seizures-list-PNxx.txt` per patient.

### 2. Convert to HDF5

```bash
python siena2hdf.py datasets/raw/siena/1.0.0 --out_hdf_root datasets/hdf/siena --n_jobs 8
```

- Channel names are normalised, and only supported channels are kept: 31 EEG electrodes, plus SpO2, heart rate and plethysmogram.
- Signals are resampled to 256 Hz and stored as float32 without further filtering.
- Seizure intervals are taken from the seizure lists and converted to seconds relative to the start of each recording.

### 3. Create the core dataset and views

```bash
python create_siena_ds.py --hdf_root datasets/hdf/siena --out_dir datasets/views/siena
```

| File | Purpose |
|---|---|
| `siena_core.h5` | core dataset |
| `siena_selfsup_dv.h5` | default self-supervised view with patient-level split (test: PN14, PN16, PN17; val: PN11, PN12, PN13) |
| `siena_selfsup_dv_bal.h5` | balanced view for fine-tuning: ictal samples plus an equal number of non-ictal samples per patient; test patients excluded |
| `siena_selfsup_16_17_dv.h5` | test view following the reference protocol (test: PN16, PN17) |

**Reproducing the fine-tuning selection.** A newly drawn balanced view depends on the random seed (`--seed`, default 42). To use the exact selection from the paper, pass the released template:

```bash
python create_siena_ds.py --hdf_root datasets/hdf/siena --out_dir datasets/views/siena \
    --bal_template views/siena_selfsup_dv_bal.h5
```

The template is checked against the sampling rates and signal lengths of your HDF5 files. This only works if step 2 was run with the default settings.

## CHB-MIT Scalp EEG Database

### 1. Obtain the data

The database is available from [PhysioNet](https://physionet.org/content/chbmit/1.0.0/) under the Open Data Commons Attribution License v1.0:

```bash
wget -r -N -c -np -nH --cut-dirs=2 -P datasets/raw/chbmit https://physionet.org/files/chbmit/1.0.0/
```

The expected layout is `<chb_root>/chbXX/chbXX_YY.edf` with one `chbXX-summary.txt` per patient.

### 2. Convert to HDF5

```bash
python chb2hdf.py datasets/raw/chbmit/1.0.0 --out_hdf_root datasets/hdf/chbmit --n_jobs 8
```

- Signals are kept at the native 256 Hz without filtering. Use `--sampling_freq` to resample.
- Seizure intervals are taken from the per-patient summary files.
- chb01 and chb21 were recorded from the same subject 1.5 years apart. Take this into account if you define your own patient-level splits.

### 3. Create the core dataset and views

```bash
python create_chb_ds.py --hdf_root datasets/hdf/chbmit --out_dir datasets/views/chbmit [--proper]
```

| File | Purpose |
|---|---|
| `chbmit_core_ds.h5` | core dataset |
| `chbmit_selfsup_10s_biot.h5` | default view, compatible with the BIOT/CBraMod reference protocol (10 s windows; train: chb01–chb20, val: chb21–chb22, test: chb23–chb24) |
| `chbmit_selfsup_10s_proper.h5` | as above, but with corrected ictal labelling (only with `--proper`) |

The default view reproduces the reference protocol, including its labelling, augmentation and file filtering. The view configuration lets you switch each of these off individually (`biot_compat`, `augment_ictal`, `apply_biot_process1_file_filter`). The differences are discussed in the NeuroNTP paper.

## TUH EEG Seizure Corpus (TUSZ)

### 1. Obtain the data

TUSZ is distributed by the Neural Engineering Data Consortium under the TUH data use agreement. [Register for access](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/) and download TUSZ v2.0.x (we used v2.0.3) into `datasets/raw/tusz/`.

The expected layout is `edf/{train,dev,eval}/<patient>/<session>/<montage>/<patient>_<session>_tNNN.{edf,csv,csv_bi}`.

### 2. Convert to HDF5

```bash
python tusz2hdf.py --tusz_edf_root datasets/raw/tusz/v2.0.3/edf --out_hdf_root datasets/hdf/tusz --n_jobs 16
```

- The EDF segments of each session are resampled to 256 Hz, stitched together, and filtered with a 0.5–100 Hz band-pass and a 60 Hz notch filter. Disable the filters with `--bandpass none` or `--notch none`.
- Sessions shorter than 300 s are skipped and listed in `skipped.csv`.
- Use `--require_same_channels` to check channel consistency across the segments of a session.

### 3. Create the core dataset and view

```bash
python create_tusz_ds.py --hdf_root datasets/hdf/tusz --out_dir datasets/views/tusz
```

| File | Purpose |
|---|---|
| `TUSZ_coreds.h5` | core dataset |
| `dv_selfsup_10s.h5` | pretraining view (10 s chunks) |

TUSZ is used for self-supervised pretraining only. During training, samples are drawn randomly from the view, so the view defines a pool of samples rather than a fixed sequence.

## General notes

- **Re-running.** The `create_*_ds.py` scripts reuse existing core dataset and default view files unless `--overwrite` is given. Derived views (the Siena balanced and test views, and the CHB-MIT proper view) are always rebuilt.
- **Moving data.** If you move the HDF5 store, re-create the core datasets and views (step 3) on the new location.
- **Licenses.** None of the datasets are included in this repository. Please follow the license terms of each source.

## Preparing your own dataset

To use your own data with NeuroNTP, follow the same three steps:

1. **Convert** your recordings to one HDF5 file per recording or session. Each file should contain a `signal` array (`n_channels × n_samples`) and the attributes `sfreq`, `signal_length`, `n_channels`, `ch_names`, `patient_id` and the record identifier. Annotations go in `annotations.{onset,duration,description}`, with times in seconds from the start of the recording.
2. **Write the index tables** `sessions.csv` (one row per file) and `annotations.csv` (one row per annotation) in the HDF5 root.
3. **Create** a core dataset and view by subclassing `EpilepsiaCoreDataset` and `EpilepsiaDatasetView`. Use `siena2hdf.py` and `create_siena_ds.py` as a template, since they are the simplest of the three pipelines.

Normalise channel names to the common scheme (`"eeg fp1"`, `"spo2"`, `"hr"`, ...) so the tokenizer can match channels across datasets.