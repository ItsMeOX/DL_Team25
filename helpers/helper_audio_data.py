"""Shared audio data utilities for multi-channel ToyCar models.

This module centralizes file parsing, spectrogram conversion, and dataset
classes so notebooks can stay concise and focused on model logic.
"""

from __future__ import annotations

import glob
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


DEFAULT_CHANNELS: Tuple[str, ...] = ("1", "2", "3", "4")


def load_all_wav_paths(
    data_base_path: str,
    cases: Sequence[str],
    folder: str = "NormalSound_IND",
) -> List[str]:
    """Collect all wav file paths from selected cases and folder."""
    all_wavs: List[str] = []
    for case in cases:
        case_folder = os.path.join(data_base_path, case, folder)
        all_wavs.extend(glob.glob(os.path.join(case_folder, "*.wav")))
    return all_wavs


def structure_wav_paths(wav_paths: Iterable[str]) -> defaultdict:
    """Group wav paths as data[case][sample_id][channel]."""
    grouped = defaultdict(lambda: defaultdict(dict))
    for wav_path in wav_paths:
        case, channel, sample_id = parse_standard_wav_filename(wav_path)
        grouped[case][sample_id][channel] = wav_path
    return grouped


def parse_standard_wav_filename(wav_path: str) -> Tuple[str, str, str]:
    """Parse standard filename format and return (case, channel, sample_id)."""
    parts = os.path.basename(wav_path).replace(".wav", "").split("_")
    case_name = parts[2]
    channel = parts[-2].replace("ch", "")
    sample_id = parts[-1]
    return case_name, channel, sample_id


def parse_anomaly_wav_filename(wav_path: str) -> Tuple[str, str, str]:
    """Parse anomaly filename and return (case, channel, anomaly_sample_id)."""
    parts = os.path.basename(wav_path).replace(".wav", "").split("_")
    case_name = parts[2]
    condition = parts[3]
    suffix = parts[-1]
    channel = parts[-2].replace("ch", "")
    sample_id = f"{condition}_{suffix}"
    return case_name, channel, sample_id


def crop_audio(
    audio: np.ndarray,
    sr: int,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> np.ndarray:
    """Crop audio by seconds. If boundaries are missing, return unchanged."""
    if start_sec is None or end_sec is None:
        return audio
    start = int(start_sec * sr)
    end = int(end_sec * sr)
    return audio[max(0, start) : min(len(audio), end)]


def normalize_minmax(spec: np.ndarray) -> np.ndarray:
    """Min-max normalize spectrogram to [0, 1]."""
    return (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)


def wav_to_logmel(
    audio: np.ndarray, 
    sr: int, 
    n_fft = 1024, 
    hop_length=512, 
    n_mels=64
) -> np.ndarray:
    """Convert waveform to normalized log-mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return normalize_minmax(log_mel)

def min_max_normalize(spec):
    return (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)

def load_audio(files):
    """Accept str or list[str], return concatenated audio"""
    if isinstance(files, str):
        audio, sr = librosa.load(files, sr=None)
        return audio, sr

    audios = []
    sr = None

    for f in files:
        audio, sr = librosa.load(f, sr=None)
        audios.append(audio)

    return np.concatenate(audios), sr

def plot_cropped_audio(
    files: str,
    title: str,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> None:
    """Plot full and optionally cropped log-Mel spectrograms."""
    
    audio, sr = load_audio(files)

    # Full spectrogram
    log_full = wav_to_logmel(audio, sr)

    # If cropping is specified
    if start_sec is not None and end_sec is not None:
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        audio_crop = audio[start_sample:end_sample]

        mel_crop = librosa.feature.melspectrogram(
            y=audio_crop, sr=sr, n_fft=1024, hop_length=512, n_mels=64
        )
        log_crop = librosa.power_to_db(mel_crop, ref=np.max)
        log_crop_norm = min_max_normalize(log_crop)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Full
        img1 = librosa.display.specshow(
            log_full, sr=sr, hop_length=512,
            x_axis="time", y_axis="mel", ax=axes[0]
        )
        axes[0].set_title("Log-Mel Spectrogram")

        # Cropped
        img2 = librosa.display.specshow(
            log_crop, sr=sr, hop_length=512,
            x_axis="time", y_axis="mel", ax=axes[1]
        )
        axes[1].set_title(f"Cropped ({start_sec}-{end_sec}s)")

        # Cropped normalized
        img3 = librosa.display.specshow(
            log_crop_norm, sr=sr, hop_length=512,
            x_axis="time", y_axis="mel", ax=axes[2],
            vmin=0, vmax=1
        )
        axes[2].set_title("Cropped (Normalized)")

        fig.suptitle(title)

        fig.colorbar(img1, ax=axes[0])
        fig.colorbar(img2, ax=axes[1])
        fig.colorbar(img3, ax=axes[2])

    else:
        # Only full
        fig, ax = plt.subplots(figsize=(6, 3))

        img = librosa.display.specshow(
            log_full, sr=sr, hop_length=512,
            x_axis="time", y_axis="mel", ax=ax
        )
        ax.set_title(f"{title} (Full)")
        fig.colorbar(img, ax=ax)

    plt.tight_layout()
    plt.show()


def plot_cnt_segmented_audio(file_paths, title, top_db=30, min_duration=3.0):
    audio_all = []
    sr = None

    # Concatenate all audios
    for path in file_paths:
        audio, sr = librosa.load(path, sr=None)
        audio_all.append(audio)
    audio_concat = np.concatenate(audio_all)
    print(f"Total duration: {len(audio_concat)/sr:.2f}s")

    log_full = wav_to_logmel(audio_concat, sr=sr)

    # Segmentation
    intervals = librosa.effects.split(audio_concat, top_db=top_db)

    segments = []
    for start, end in intervals:
        duration = (end - start) / sr
        start_sec = start / sr
        end_sec = end / sr
        
        if duration < min_duration:
            continue

        print(f"Segment: {start_sec:.2f}s -> {end_sec:.2f}s | Duration: {duration:.2f}s")
        segments.append(audio_concat[start:end])

    print(f"Kept {len(segments)} segments")

    # Concatenate only sound segments
    if len(segments) > 0:
        audio_segments = np.concatenate(segments)
    else:
        audio_segments = np.array([])

    # Segment logmel
    if len(audio_segments) > 0:
        log_seg = wav_to_logmel(audio_segments, sr=sr)
    else:
        log_seg = None

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # full
    img1 = librosa.display.specshow(
        log_full,
        sr=sr,
        hop_length=512,
        x_axis='time',
        y_axis='mel',
        ax=axes[0]
    )
    axes[0].set_title("Concatenated (Raw with Silence)")

    # segments
    if log_seg is not None:
        img2 = librosa.display.specshow(
            log_seg,
            sr=sr,
            hop_length=512,
            x_axis='time',
            y_axis='mel',
            ax=axes[1]
        )
        axes[1].set_title("Extracted Segments (Silence Removed)")
    else:
        axes[1].set_title("No segments found")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def extract_and_save_cnt_segments(
    channel_paths: Dict[str, List[str]],
    save_path: str,
    segment_sec: float = 600.0,
    skip_sec: float = 1200.0,
    start_offset_sec: float = 0.0,
    top_db: int = 30,
    min_len_sec: float = 1.0,
    plot: bool = False, 
    max_plots: int = 5
) -> None:
    """
    Extract fixed-length non-silent segments from CNT audio and save as .npy files.

    The function:
    - Concatenates multi-file audio streams per channel
    - Extracts segments with fixed length
    - Removes silent regions
    - Converts to log-Mel spectrogram
    - Saves as multi-channel tensors
    """

    # Create output directory
    os.makedirs(save_path, exist_ok=True)

    # Load one file to get sample rate and length
    # (Assumes all files have same sampling rate and duration)
    sample_audio, sr = librosa.load(channel_paths['1'][0], sr=None)
    file_len = len(sample_audio)

    # Convert segment duration (sec) -> samples
    segment_len = int(segment_sec * sr)
    skip_len = int(skip_sec * sr)

    # Total number of files per channel
    num_files = len(channel_paths['1'])

    # Helper fn
    def remove_silence(audio: np.ndarray) -> np.ndarray | None:
        intervals = librosa.effects.split(audio, top_db=top_db)
        min_len = int(min_len_sec * sr)

        segments = [
            audio[start:end]
            for start, end in intervals
            if (end - start) >= min_len
        ]

        return np.concatenate(segments) if segments else None

    # Start extraction here
    global_ptr = int(start_offset_sec * sr)     # pointer over the entire concatenated stream, offset by start_offset_sec initially
    count = 0                                   # segment counter

    # Total length across all files (in samples)
    max_len = num_files * file_len

    # Continue until we can no longer extract a full segment
    while global_ptr + segment_len <= max_len:

        mel_channels = []

        # Process each channel separately
        for ch in ['1', '2', '3', '4']:
            remaining = segment_len  # how much audio still needed
            ptr = global_ptr         # local pointer for this channel
            chunks = []              # pieces of audio to reconstruct segment

            # Reconstruct segment across files
            while remaining > 0:
                file_idx = ptr // file_len   # which file we are in
                offset = ptr % file_len      # position inside that file

                audio, _ = librosa.load(channel_paths[ch][file_idx], sr=None)

                # Number of samples we can take from this file
                take = min(remaining, file_len - offset)

                # Extract chunk and store
                chunks.append(audio[offset:offset + take])

                # Move pointer forward
                ptr += take
                remaining -= take

            # Combine chunks into one continuous segment
            segment = np.concatenate(chunks)

            # Remove silence from segment
            segment = remove_silence(segment)
            # If completely silent, skip this channel
            if segment is None:
                continue

            # Ensure fixed segment length
            if len(segment) < segment_len:
                # Pad with zeros if too short
                segment = np.pad(segment, (0, segment_len - len(segment)))
            else:
                # Truncate if too long
                segment = segment[:segment_len]

            mel = wav_to_logmel(segment, sr)
            mel_channels.append(mel)

        # Ensure all 4 channels exist
        if len(mel_channels) < 4:
            global_ptr += segment_len + skip_len
            continue

        # Align time dimension across channels
        # (due to slight differences after processing)
        min_T = min(m.shape[1] for m in mel_channels)
        mel_channels = [m[:, :min_T] for m in mel_channels]

        # Stack into shape: (4, Mel, Time)
        x = np.stack(mel_channels)

        # Plot optionally
        if plot and count < max_plots:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            for ch_idx in range(4):
                img = librosa.display.specshow(
                    x[ch_idx],
                    x_axis='time',
                    y_axis='mel',
                    sr=sr,
                    hop_length=512,
                    ax=axes[ch_idx],
                    vmin=0, vmax=1
                )
                axes[ch_idx].set_title(f"Segment {count} - Ch{ch_idx+1}")

            fig.colorbar(img, ax=axes)
            plt.tight_layout()
            plt.show()

        # Save
        np.save(os.path.join(save_path, f"{count:05d}.npy"), x)

        print(f"[SEGMENT {count}] at {global_ptr/sr:.2f}s")

        # Move to next segment (with skip)
        count += 1
        global_ptr += segment_len + skip_len

    print(f"[INFO] Done. Extracted {count} segments.")


def build_multichannel_tensor(
    channel_to_path: Dict[str, str],
    channels: Sequence[str] = DEFAULT_CHANNELS,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> torch.Tensor:
    """Load multi-channel audio, convert each channel to log-Mel spectrogram,
    align their time dimensions, and stack into a single tensor."""
    
    mel_channels: List[np.ndarray] = []
    for channel in channels:
        audio, sr = librosa.load(channel_to_path[channel], sr=None)
        
        # Crop audio to the specified time window (if provided)
        # Ensures consistent segment length across all samples
        audio = crop_audio(audio, sr, start_sec=start_sec, end_sec=end_sec)

        # Convert waveform -> log-Mel spectrogram
        # This transforms raw audio into time-frequency representation
        mel = wav_to_logmel(audio, sr)

        mel_channels.append(mel)

    # Align time dimension across channels
    # Due to minor differences (e.g., rounding, cropping),
    # spectrograms may have slightly different time lengths.
    # We truncate all to the shortest one to ensure alignment.
    min_time = min(mel.shape[1] for mel in mel_channels)
    mel_channels = [mel[:, :min_time] for mel in mel_channels] 

    # Stack into multi-channel (4 channels for ToyCar dataset) tensor
    # Final shape: (C, Mel, Time)
    return torch.tensor(np.stack(mel_channels), dtype=torch.float32)


def group_wavs(
    data_base_path: str,
    cases: Sequence[str],
    folder_name: str,
    channels: Sequence[str] = DEFAULT_CHANNELS,
    anomaly_mode: bool = False,
) -> List[Tuple[Tuple[str, str], Dict[str, str]]]:
    """Return valid grouped sample list: [((case, sample_id), {ch: path})]."""
    grouped: Dict[Tuple[str, str], Dict[str, str]] = {}
    for case in cases:
        folder_path = os.path.join(data_base_path, case, folder_name)
        for wav_path in glob.glob(os.path.join(folder_path, "*.wav")):
            if anomaly_mode:
                case_name, channel, sample_id = parse_anomaly_wav_filename(wav_path)
            else:
                case_name, channel, sample_id = parse_standard_wav_filename(wav_path)

            key = (case_name, sample_id)
            if key not in grouped:
                grouped[key] = {}
            grouped[key][channel] = wav_path

    grouped_samples: List[Tuple[Tuple[str, str], Dict[str, str]]] = []
    for key, channel_map in grouped.items():
        if all(ch in channel_map for ch in channels):
            grouped_samples.append((key, channel_map))
    return grouped_samples


class GroupedAudioDataset(Dataset):
    """Base dataset for grouped 4-channel ToyCar spectrogram samples."""

    def __init__(
        self,
        grouped_samples: Sequence[Tuple[Tuple[str, str], Dict[str, str]]],
        channels: Sequence[str] = DEFAULT_CHANNELS,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
    ) -> None:
        self.channels = list(channels)
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.sample_keys = [key for key, _ in grouped_samples]
        self.samples = [channel_map for _, channel_map in grouped_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        channel_map = self.samples[idx]
        try:
            return build_multichannel_tensor(
                channel_map,
                channels=self.channels,
                start_sec=self.start_sec,
                end_sec=self.end_sec,
            )
        except Exception as exc:  # defensive fallback for corrupted files
            print(f"[ERROR] Failed sample {self.sample_keys[idx]}: {exc}")
            return self.__getitem__((idx + 1) % len(self.samples))


class NormalIndDataset(GroupedAudioDataset):
    """4-channel individual (IND) normal dataset from `NormalSound_IND` wav files."""

    def __init__(
        self,
        data_base_path: str,
        cases: Sequence[str],
        channels: Sequence[str] = DEFAULT_CHANNELS,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
        verbose: bool = True,
    ) -> None:
        grouped = group_wavs_for_dataset(
            data_base_path=data_base_path,
            cases=cases,
            folder_name="NormalSound_IND",
            channels=channels,
            anomaly_mode=False,
        )
        super().__init__(grouped, channels=channels, start_sec=start_sec, end_sec=end_sec)
        if verbose:
            print(f"[INFO] Normal grouped samples: {len(self.samples)}")


class AnomalousIndDataset(GroupedAudioDataset):
    """4-channel individual (IND) anomalous dataset from `AnomalousSound_IND` wav files."""

    def __init__(
        self,
        data_base_path: str,
        cases: Sequence[str],
        channels: Sequence[str] = DEFAULT_CHANNELS,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
        verbose: bool = True,
    ) -> None:
        grouped = group_wavs_for_dataset(
            data_base_path=data_base_path,
            cases=cases,
            folder_name="AnomalousSound_IND",
            channels=channels,
            anomaly_mode=True,
        )
        super().__init__(grouped, channels=channels, start_sec=start_sec, end_sec=end_sec)
        if verbose:
            print(f"[INFO] Anomalous grouped samples: {len(self.samples)}")
