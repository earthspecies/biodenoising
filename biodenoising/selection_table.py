import glob
import os
from typing import List, Optional, Sequence, Tuple

import pandas as pd
import torch


def _normalize_column_name(name: str) -> str:
    """
    Normalize a column name for fuzzy matching.
    Lowercase, strip, and remove spaces/underscores for robust comparisons.
    """
    norm = name.lower().strip()
    norm = norm.replace(" ", "").replace("_", "")
    return norm


def find_selection_table_for(audio_filepath: str) -> Optional[str]:
    """
    Look for a csv/tsv/txt file in the same directory as the audio file whose
    filename contains the audio stem.

    Parameters
    ----------
    audio_filepath : str
        Full path to the audio file.

    Returns
    -------
    Optional[str]
        Path to the first matching table file, or None if none found.
    """
    directory = os.path.dirname(audio_filepath)
    stem = os.path.splitext(os.path.basename(audio_filepath))[0]
    candidates: List[str] = []
    for ext in ("csv", "tsv", "txt"):
        pattern = os.path.join(directory, f"*{stem}*.{ext}")
        candidates.extend(glob.glob(pattern))
    return candidates[0] if candidates else None


def load_events_seconds(table_path: Optional[str]) -> List[Tuple[float, float]]:
    """
    Load selection table and extract start/end times in seconds.
    Tries to match columns for start and end using fuzzy rules.

    Parameters
    ----------
    table_path : Optional[str]
        Path to the table file (csv, tsv, or txt). If None, returns empty list.

    Returns
    -------
    List[Tuple[float, float]]
        A list of (start_seconds, end_seconds) tuples.
    """
    if table_path is None:
        return []

    try:
        df = pd.read_csv(table_path, sep=None, engine="python")
    except Exception:
        if table_path.endswith((".tsv", ".txt")):
            df = pd.read_csv(table_path, sep="\t")
        else:
            df = pd.read_csv(table_path)

    normalized = {col: _normalize_column_name(col) for col in df.columns}

    start_aliases = {"start", "beginning", "begintime", "begin"}
    end_aliases = {"end", "endtime"}

    start_col: Optional[str] = None
    end_col: Optional[str] = None
    for col, norm in normalized.items():
        if start_col is None and (norm == "start" or any(alias in norm for alias in start_aliases)):
            start_col = col
        if end_col is None and (norm == "end" or any(alias in norm for alias in end_aliases)):
            end_col = col
        if start_col is not None and end_col is not None:
            break

    if start_col is None or end_col is None:
        return []

    try:
        starts = pd.to_numeric(df[start_col], errors="coerce").astype(float).tolist()
        ends = pd.to_numeric(df[end_col], errors="coerce").astype(float).tolist()
    except Exception:
        return []

    events: List[Tuple[float, float]] = []
    for s, e in zip(starts, ends):
        if pd.isna(s) or pd.isna(e):
            continue
        if e <= s:
            continue
        events.append((float(s), float(e)))
    return events


def build_mask_from_events(
    length_frames: int,
    sample_rate: int,
    events: Sequence[Tuple[float, float]],
    device: torch.device | str,
) -> torch.Tensor:
    """
    Build a 1D mask tensor of shape [length_frames] with ones inside any event
    intervals and zeros elsewhere.

    Parameters
    ----------
    length_frames : int
        Total number of frames (samples) in the signal.
    sample_rate : int
        Sample rate of the signal.
    events : Sequence[Tuple[float, float]]
        Event intervals in seconds.
    device : torch.device | str
        Torch device for the output tensor.

    Returns
    -------
    torch.Tensor
        A 1D tensor mask of length `length_frames`.
    """
    if not events:
        return torch.ones(length_frames, device=device)

    mask = torch.zeros(length_frames, device=device)
    for start_s, end_s in events:
        start_idx = int(max(0, round(start_s * sample_rate)))
        end_idx = int(min(length_frames, round(end_s * sample_rate)))
        if end_idx > start_idx:
            mask[start_idx:end_idx] = 1.0
    if mask.sum().item() == 0:
        mask[:] = 1.0
    return mask



