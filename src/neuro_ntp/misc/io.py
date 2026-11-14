"""
+----------------------------------------------------------------------------------------------------------------------+
io.py
EDF and HDF5 file I/O utilities

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Authors: Philipp Seidl (philipp.seidl@jku.at), Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+

Description:
    Low-level reading of EDF recordings and HDF5 signal stores, including partial reads of the signal by sample range.

        load_edf             full EDF read with MNE (optional resampling and re-referencing)
        load_partial_edf     fast partial EDF read with pyEDFlib (by time range)
        get_edf_info         EDF header information without reading the signal
        load_partial_hdf5    partial HDF5 read by sample range, with optional channel selection and ordering; used
                             by the dataset views to retrieve the signal of a sample
        unmix_channels       reorder signal channels to a given channel order
+----------------------------------------------------------------------------------------------------------------------+
"""

import time
import random
from contextlib import contextmanager

import h5py
import mne
import numpy as np
import pyedflib

import logging
logger = logging.getLogger(__name__)


def unmix_channels(signal, current_ch_order, ch_order, verbose=False):
    """
    reorder channels in signal based on ch_order
    signal: (n_channels, n_samples) numpy array
    current_ch_order: list of channel names
    ch_order: list of channel names
    """
    if signal is None: 
        return signal
    # if it is an array make it a list
    if isinstance(ch_order, np.ndarray):
        ch_order = ch_order.tolist()
    # if mixed up channels
    if signal.shape[0] != len(ch_order):
        new_signal = np.zeros([len(ch_order), signal.shape[1]])
        # get corresponding channels
        for i, ch_name in enumerate(ch_order):
            if ch_name in current_ch_order:
                new_signal[i, :] = signal[current_ch_order.index(ch_name), :]
        signal = new_signal

    return signal


def load_edf(edf_file, new_sampling_freq=None, set_reference=True, verbose=False):
    """
    Load an EDF file using MNE.
    same result as load_partial_edf

    Returns:
    - A dictionary containing the following:
        - ch_names: A list of channel names.
        - signal: A 2D numpy array containing the signal data.
        - info: A dictionary containing additional information.
        - annotations: A dictionary containing the annotations.
    """
    try:
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False, encoding='latin1')
    except Exception as e:
        logger.error(f"Error reading {edf_file} as EDF file. Skipping.")
        return None
    if verbose: 
        logger.debug("reading EDF file completed")
    
    # Resample if a new sampling frequency is provided
    if new_sampling_freq is not None and new_sampling_freq != raw.info['sfreq']:
        raw.resample(new_sampling_freq)
        # throw warning if the new sampling frequency is lower than the original
        if new_sampling_freq < raw.info['sfreq']:
            logger.warn("Warning: new sampling frequency is lower than original")
        if verbose: logger.debug(f"resampling completed: from {raw.info['sfreq']} to {new_sampling_freq}")

    if set_reference:
        if 'EEG M1' in raw.ch_names and 'EEG M2' in raw.ch_names:
            raw = raw.set_eeg_reference(ref_channels=['EEG M1', 'EEG M2'], ch_type='eeg', verbose=verbose)
        else:
            logger.info(f" {edf_file} has no M1 and M2 channels, using average reference")
            raw = raw.set_eeg_reference(ref_channels='average', ch_type='eeg', verbose=verbose) # consider using CZ as ref channel

    results = {
        'ch_names': raw.ch_names,
        'signal': raw.get_data(),
        'info': {
            'meas_date': str(raw.info['meas_date']),
            'lowpass': raw.info['lowpass'],
            'highpass': raw.info['highpass'],
            'sfreq': raw.info['sfreq'], # changed to new_sampling_freq if resampled
            'bads': raw.info['bads'],
            'subject_info': raw.info['subject_info'],
        },
        'annotations': {
            'onset': raw.annotations.onset,
            'duration': raw.annotations.duration,
            'description': raw.annotations.description,
        },
    }
    return results

def get_edf_info(edf_file):
    """
    Get information from an EDF file.
    
    Parameters:
    - edf_file: Path to the EDF file.
    
    Returns:
    - A dictionary containing the following:
        - file_duration: The duration of the EDF file in seconds.
        - annotations: A list of annotations.
        - n_signals: The number of signals in the EDF file.
        - signal_labels: A list of signal labels.
        - sample_rates: A list of sample rates.
        - n_samples: A list of the number of samples for each signal.
        - start_datetime: The start date and time of the recording.
        - patient_id: The patient ID.
        - patient
    """
    with pyedflib.EdfReader(edf_file) as f:
        file_duration = f.file_duration
        n_signals = f.signals_in_file
        annotations = f.readAnnotations()
        signal_labels = f.getSignalLabels()
        sample_rates = f.getSampleFrequencies()
        n_samples = f.getNSamples()
        start_datetime = f.getStartdatetime()
        patient_id = f.getPatientCode()
        patient_name = f.getPatientName()
        patient_additional = f.getPatientAdditional()
        patient_birthdate = f.getPatientBirthdate()
        patient = {"patient_id": patient_id, "patient_name": patient_name, "patient_additional": patient_additional, "patient_birthdate": patient_birthdate}
    return {"file_duration": file_duration, "annotations": annotations, "n_signals": n_signals, "signal_labels": signal_labels, "sample_rates": sample_rates, "n_samples": n_samples, "start_datetime": start_datetime, "patient": patient}

def load_partial_edf(filepath, start_sec=0, stop_sec=None):
    """
    Load a partial EDF file FAST using pyEDFlib.
    
    Parameters:
    - filepath: Path to the EDF file.
    - start_sec: Start time in seconds.
    - stop_sec: Stop time in seconds. If None, the entire signal is loaded. If 0, --> very fast, e.g. for getting channel names and annotations.

    Returns:
    - A dictionary containing the following
        - ch_names: A list of channel names.
        - data: A 2D numpy array containing the signal data.
        - annotations: A list of annotations.
        - info: A dictionary containing additional information.

    """
    results = {'ch_names': None, 'data': None, 'annotations': None, 'info': {}}
    with pyedflib.EdfReader(filepath) as f:
        # Get sampling rate of the first signal
        sample_rate = f.getSampleFrequency(0)
        
        # Calculate start and stop indices
        start_idx = int(start_sec * sample_rate)
        stop_idx = f.getNSamples()[0] if stop_sec is None else int(stop_sec * sample_rate)
        
        # Read and select a segment for all signals
        n = f.signals_in_file
        signal_labels = f.getSignalLabels()

        data = np.zeros((n, stop_idx-start_idx))
        for i in range(n):
            data[i, :] = f.readSignal(i, start=start_idx, n=stop_idx-start_idx)
        results['ch_names'] = signal_labels
        results['data'] = data
        results['info']['meas_date'] = f.getStartdatetime()
        results['info']['sample_rate'] = sample_rate
        results['info']['n_samples'] = stop_idx - start_idx
        results['annotations'] = f.readAnnotations()

    return results

def load_edf_full(filepath):
    return load_partial_edf(filepath, start_sec=0, stop_sec=None)


@contextmanager
def _open_h5(path, mode="r", swmr=True):
    # Safer open on network FS
    f = h5py.File(path, mode, libver="latest", swmr=swmr)
    try:
        yield f
    finally:
        f.close()


def load_partial_hdf5(hdf_path, start_idx=0, stop_idx=None, step_size=1, only_signal=False, channel_order=None,
                      load_new_annotations=False, load_additional_sensors=False, max_retries=5, base_sleep=0.15):
    """
    Load a partial HDF5 file.

    Parameters:
    - hdf_path: Path to the HDF5 file.
    - start_idx: Start index.
    - stop_idx: Stop index. If None, the entire signal is loaded.
    - step_size: Step size. Default is 1.
    - only_signal: If True, only the signal data is loaded.
    - signal_channels: List of channel names in the desired order. If None, all channels are used.
    - load_new_annotations: If True, new annotations are loaded under annotations2 (if available). (alternative annotation set, if present in the file)
    - load_additional_sensors: If True, additional sensors are loaded under additional_sensors (if available). - signal_* and ch_names_* are loaded and concatenated with the original signal.

    Returns:
    - A dictionary containing the following:
        - ch_names: A list of channel names.
        - signal: A 2D numpy array containing the signal data.
        - info: A dictionary containing additional information.
        - annotations: A dictionary containing the annotations.
    """

    results = {}
    for attempt in range(max_retries):
        try:
            with _open_h5(hdf_path, "r", swmr=True) as f:

                annotations = {
                    'onset': f['annotations.onset'][:] if 'annotations.onset' in f.keys() else None,
                    'duration': f['annotations.duration'][:] if 'annotations.duration' in f.keys() else None,
                    'description': np.array([a.decode('utf-8') for a in f['annotations.description'][
                                                                        :]]) if 'annotations.description' in f.keys() else None,
                }
                if load_new_annotations:
                    annotations = {
                        'onset': f['annotations2.onset'][:] if 'annotations2.onset' in f.keys() else f['annotations.onset'][
                                                                                                     :],
                        'duration': f['annotations2.duration'][:] if 'annotations2.duration' in f.keys() else f[
                                                                                                                  'annotations.duration'][
                                                                                                              :],
                        'description': np.array([a.decode('utf-8') for a in f['annotations2.description'][
                                                                            :]]) if 'annotations2.description' in f.keys() else np.array(
                            [a.decode('utf-8') for a in f['annotations.description'][:]]),
                    }

                info = {k: v for k, v in f.attrs.items() if k != 'signal'}

                results['info'] = info
                if 'ch_names' in info.keys():
                    results['ch_names'] = info['ch_names']
                elif 'ch_names' in f.keys():
                    results['ch_names'] = f['ch_names'][:].tolist()
                else:
                    results['ch_names'] = []
                results['annotations'] = annotations.copy()

                results['ch_names'] = np.array([ch_name.lower() for ch_name in results['ch_names']])
                if channel_order:
                    channel_order = ([ch_name.lower() for ch_name in channel_order])

                additional_sensor = f.keys()
                additional_sensor = [ch for ch in additional_sensor if 'signal_' in ch]  # filter by signal_

                # only load intersection
                mask = [ch in channel_order for ch in results['ch_names']] if channel_order else np.ones(
                    len(results['ch_names']), dtype=bool)
                if stop_idx is None or stop_idx < 0:
                    stop_idx = info['signal_length']
                signal = f['signal'][mask, start_idx:stop_idx:step_size] if 'signal' in f.keys() else np.array([])
                results['ch_names'] = results['ch_names'][mask] if 'ch_names' in info.keys() else np.array([])
                results['sensor'] = ['EEG'] * len(results['ch_names']) if 'ch_names' in info.keys() else []

                if load_additional_sensors:
                    for ch in additional_sensor:
                        add_ch_names = f[f'ch_names_{ch.replace("signal_", "")}'][:]
                        add_ch_names = np.array([a.decode('utf-8').lower() for a in add_ch_names])
                        mask = [ch in channel_order for ch in add_ch_names] if channel_order else np.ones(len(add_ch_names),
                                                                                                          dtype=bool)
                        add_signale = f[ch][start_idx:stop_idx:step_size, mask]
                        add_signale = add_signale.T
                        if signal is None:
                            signal = add_signale
                        else:
                            signal = np.concatenate([signal, add_signale], axis=0) if signal.shape[0] != 0 else add_signale
                        results['ch_names'] = np.concatenate([results['ch_names'], add_ch_names[mask]]) if \
                        results['ch_names'].shape[0] != 0 else add_ch_names[mask]
                        results['sensor'] += [ch.replace("signal_", "")] * len(add_ch_names[mask])

                # if signal-length not start_idx-stop_idx at a certain step-size
                if signal.shape[1] != (((stop_idx - start_idx) // step_size)):
                    logger.warn(f'Warning: signal length not equal to start_idx-stop_idx: {signal.shape[1]} != {(stop_idx - start_idx) // step_size}, 0-padding')
                    new_signal = np.zeros([signal.shape[0], (((stop_idx - start_idx) // step_size) + 1)])
                    new_signal[:, :signal.shape[1]] = signal
                    signal = new_signal

                results['signal'] = unmix_channels(signal, results['ch_names'].tolist(),
                                                   channel_order) if channel_order else signal
                results['sensor'] = np.array(results['sensor'])

            return results

        except (OSError, IOError, KeyError) as e:
            last_err = e
            time.sleep(base_sleep * (2 ** attempt) + random.uniform(0, 0.05))
            logger.error(f"[ERROR] HDF5 file reading error - retrying... attempt: {attempt}")

    logger.error(f"[ERROR] An error occurred: {last_err}")
    raise last_err

