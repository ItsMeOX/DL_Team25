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


def build_multichannel_tensor(
    channel_to_path: Dict[str, str],
    channels: Sequence[str] = DEFAULT_CHANNELS,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> torch.Tensor:
    """Load and convert all channels, then align time and stack to tensor."""
    mel_channels: List[np.ndarray] = []
    for channel in channels:
        audio, sr = librosa.load(channel_to_path[channel], sr=None)
        audio = crop_audio(audio, sr, start_sec=start_sec, end_sec=end_sec)
        mel_channels.append(wav_to_logmel(audio, sr))

    min_time = min(mel.shape[1] for mel in mel_channels)
    mel_channels = [mel[:, :min_time] for mel in mel_channels]
    return torch.tensor(np.stack(mel_channels), dtype=torch.float32)


def group_wavs_for_dataset(
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


class NormalDataset(GroupedAudioDataset):
    """4-channel normal dataset from `NormalSound_IND` wav files."""

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


class AnomalousDataset(GroupedAudioDataset):
    """4-channel anomalous dataset from `AnomalousSound_IND` wav files."""

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
