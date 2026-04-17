"""Reusable .npy preprocessing and dataset utilities."""

from __future__ import annotations

import glob
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from helpers.helper_audio_data import wav_to_logmel


def convert_to_logmel(
    structured_paths: Dict[str, Dict[str, Dict[str, str]]],
    channels: Sequence[str],
) -> defaultdict:
    """Convert grouped wav path map into log-mel tensors per sample."""
    mel_data = defaultdict(dict)
    for case, samples in structured_paths.items():
        for sample_id, channel_map in samples.items():
            if not all(ch in channel_map for ch in channels):
                continue
            mel_channels = []
            for ch in channels:
                audio, sr = librosa.load(channel_map[ch], sr=None)
                mel_channels.append(wav_to_logmel(audio, sr))
            mel_data[case][sample_id] = np.stack(mel_channels)
    return mel_data


def _load_npy_task(args: Tuple[str, str, str]) -> Tuple[str, str, np.ndarray]:
    case, file_path, sample_id = args
    return case, sample_id, np.load(file_path, mmap_mode="r")


def load_precomputed_samples(folder: str, case_limit: Optional[int] = None) -> defaultdict:
    """Load grouped `.npy` samples as data[case][sample_id] in parallel."""
    mel_data = defaultdict(dict)
    tasks: List[Tuple[str, str, str]] = []

    case_names = [c for c in os.listdir(folder) if os.path.isdir(os.path.join(folder, c))]
    if case_limit is not None:
        case_names = case_names[:case_limit]

    for case in case_names:
        case_path = os.path.join(folder, case)
        for file_name in os.listdir(case_path):
            if file_name.endswith(".npy"):
                sample_id = file_name[:-4]
                file_path = os.path.join(case_path, file_name)
                tasks.append((case, file_path, sample_id))

    with ThreadPoolExecutor(max_workers=16) as executor:
        for case, sample_id, mel in executor.map(_load_npy_task, tasks):
            mel_data[case][sample_id] = mel

    return mel_data


def reduce_samples(structured_samples: Dict[str, Dict], max_per_case: int = 200) -> Dict[str, Dict]:
    """Randomly subsample each case to control data volume."""
    reduced: Dict[str, Dict] = {}
    for case, sample_map in structured_samples.items():
        samples = list(sample_map.items())
        selected = random.sample(samples, min(max_per_case, len(samples)))
        reduced[case] = dict(selected)
    return reduced


def sec_to_frame(sec: float, sr: int, hop_length: int = 512) -> int:
    """Convert seconds into frame index for mel spectrogram timeline."""
    return int(sec * sr / hop_length)


def crop_mel(
    mel: np.ndarray,
    start_sec: float,
    end_sec: float,
    sr: int = 48000,
    hop_length: int = 512,
) -> np.ndarray:
    """Crop mel tensor with shape (channels, mels, time) by second range."""
    start_frame = sec_to_frame(start_sec, sr=sr, hop_length=hop_length)
    end_frame = sec_to_frame(end_sec, sr=sr, hop_length=hop_length)
    start_frame = max(0, start_frame)
    end_frame = min(mel.shape[-1], end_frame)
    return mel[:, :, start_frame:end_frame]


def split_cnt_to_segments_and_save(
    data_base_path: str,
    max_workers: int = 8,
) -> None:
    """
    Split CNT spectrogram `.npy` files into fixed chunks that have the same length as IND spectrogram.
    Save the result `.npy` files into `cnt_seg_root` for further use.
    """
    cnt_root = os.path.join(data_base_path, "npy", "CNT")
    ind_root = os.path.join(data_base_path, "npy", "IND")
    cnt_seg_root = os.path.join(data_base_path, "npy", "CNT_SEG")
    os.makedirs(cnt_seg_root, exist_ok=True)

    sample_ind_files = glob.glob(os.path.join(ind_root, "*", "normal", "*.npy"))
    if not sample_ind_files:
        raise FileNotFoundError("No IND normal files found.")

    sample_ind_data = np.load(sample_ind_files[0]) # Shape: (n_channels, n_mels, n_timeframes)
    time_axis = -1
    ind_frame_count = sample_ind_data.shape[time_axis]
    print(f"[INFO] IND length = {ind_frame_count} frames")

    def process_file(file_path: str):
        try:
            base_name = os.path.basename(file_path).replace(".npy", "")
            if "_seg" in base_name:
                return None

            spectrogram = np.load(file_path)
            total_frames = spectrogram.shape[time_axis]
            num_full_chunks = total_frames // ind_frame_count
            remainder = total_frames % ind_frame_count

            case_name = os.path.basename(os.path.dirname(file_path))
            case_output_dir = os.path.join(cnt_seg_root, case_name)
            os.makedirs(case_output_dir, exist_ok=True)

            saved = 0
            for i in range(num_full_chunks):
                start = i * ind_frame_count
                end = start + ind_frame_count
                slices = [slice(None)] * spectrogram.ndim
                slices[time_axis] = slice(start, end)
                segment = spectrogram[tuple(slices)]
                output_path = os.path.join(case_output_dir, f"{base_name}_seg{i:02d}.npy")
                np.save(output_path, segment)
                saved += 1

            return case_name, base_name, saved, remainder
        except Exception as exc:
            print(f"[ERROR] {file_path}: {exc}")
            return None

    cnt_files = glob.glob(os.path.join(cnt_root, "*", "*.npy"))
    print(f"[INFO] Found {len(cnt_files)} CNT files")

    # Use multi-threading workers for parallelization and speedup
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, file_path) for file_path in cnt_files]
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            case_name, base_name, saved, remainder = result
            if remainder > 0:
                print(f"  -> {case_name}/{base_name}: {saved} chunks, discarded {remainder}")
            else:
                print(f"  -> {case_name}/{base_name}: {saved} chunks")

    print(f"[SUCCESS] Done. Saved in {cnt_seg_root}")


class UnifiedNPYDataset(Dataset):
    """Dataset over IND/CNT `.npy` spectrogram files with optional balancing."""

    def __init__(
        self,
        base_path: str,
        target_T: int,
        cases: Sequence[str] = ("case1",),
        use_ind: bool = True,
        use_cnt: bool = True,
        include_anomaly: bool = True,
        stride_ratio: float = 5.0,
        seed: int = 42,
    ) -> None:
        self.samples: List[Tuple[str, Optional[int], int]] = []
        self.target_T = target_T
        self.stride = max(1, int(target_T * stride_ratio))
        self.rng = np.random.default_rng(seed)

        ind_normal_samples: List[Tuple[str, Optional[int], int]] = []
        ind_anom_samples: List[Tuple[str, Optional[int], int]] = []
        cnt_candidates: List[Tuple[str, Optional[int], int]] = []
        dropped_short = 0

        for case in cases:
            if use_ind:
                normal_dir = os.path.join(base_path, "IND", case, "normal")
                for p in glob.glob(os.path.join(normal_dir, "*.npy")):
                    x = np.load(p, mmap_mode="r")
                    if x.shape[-1] < target_T:
                        dropped_short += 1
                        continue
                    ind_normal_samples.append((p, None, 0))

            if use_ind and include_anomaly:
                anom_dir = os.path.join(base_path, "IND", case, "anomaly")
                for p in glob.glob(os.path.join(anom_dir, "*.npy")):
                    x = np.load(p, mmap_mode="r")
                    if x.shape[-1] < target_T:
                        dropped_short += 1
                        continue
                    ind_anom_samples.append((p, None, 1))

            if use_cnt:
                cnt_seg_dir = os.path.join(base_path, "CNT_SEG", case)
                cnt_dir = os.path.join(base_path, "CNT", case)
                if os.path.isdir(cnt_seg_dir):
                    cnt_paths = glob.glob(os.path.join(cnt_seg_dir, "*.npy"))
                else:
                    seg_paths = glob.glob(os.path.join(cnt_dir, "*_seg*.npy"))
                    cnt_paths = seg_paths if seg_paths else glob.glob(os.path.join(cnt_dir, "*.npy"))

                for p in cnt_paths:
                    x = np.load(p, mmap_mode="r")
                    T = x.shape[-1]
                    if T < target_T:
                        dropped_short += 1
                        continue
                    for start in range(0, T - target_T + 1, self.stride):
                        cnt_candidates.append((p, start, 0))

        if use_ind:
            self.samples.extend(ind_normal_samples)

        sampled_cnt: List[Tuple[str, Optional[int], int]] = []
        if use_cnt:
            target_cnt = len(ind_normal_samples) if use_ind else len(cnt_candidates)
            if target_cnt > 0 and cnt_candidates:
                if len(cnt_candidates) >= target_cnt:
                    pick_idx = self.rng.choice(len(cnt_candidates), size=target_cnt, replace=False)
                    sampled_cnt = [cnt_candidates[i] for i in pick_idx]
                else:
                    sampled_cnt = cnt_candidates
                    print(f"[WARN] CNT candidates ({len(cnt_candidates)}) < target ({target_cnt})")
                self.samples.extend(sampled_cnt)

        if use_ind and include_anomaly:
            self.samples.extend(ind_anom_samples)

        print(f"[INFO] IND normal: {len(ind_normal_samples)}")
        print(f"[INFO] CNT candidates (windows): {len(cnt_candidates)}")
        print(f"[INFO] CNT sampled: {len(sampled_cnt)}")
        print(f"[INFO] IND anomaly: {len(ind_anom_samples)}")
        print(f"[WARN] Dropped short samples (< target_T): {dropped_short}")
        print(f"[INFO] Total samples: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, start, label = self.samples[idx]
        x = np.load(path, mmap_mode="r")
        if start is None: # CNT sample, crop to match IND length
            T = x.shape[-1]
            start = int(self.rng.integers(0, T - self.target_T + 1))
        x = x[:, :, start : start + self.target_T]
        return torch.from_numpy(x.copy()).float(), label


def compute_target_T_from_npy(base_path: str, cases: Sequence[str], max_samples: int = 100) -> int:
    """Estimate target time length from median IND-normal sample length."""
    lengths: List[int] = []
    for case in cases:
        normal_dir = os.path.join(base_path, "IND", case, "normal")
        paths = glob.glob(os.path.join(normal_dir, "*.npy"))
        for p in paths[:max_samples]:
            x = np.load(p, mmap_mode="r")
            lengths.append(x.shape[-1])

    if not lengths:
        raise ValueError("No IND normal .npy files found to compute target_T.")

    target_T = int(np.median(lengths))
    print(f"[INFO] target_T from IND: {target_T}")
    print(f"[INFO] min={min(lengths)}, max={max(lengths)}")
    return target_T


def split_indices(indices: Sequence[int], train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Split an index list into train/val/test by ratios."""
    n = len(indices)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)
    train = list(indices[:train_end])
    val = list(indices[train_end:val_end])
    test = list(indices[val_end:])
    return train, val, test
