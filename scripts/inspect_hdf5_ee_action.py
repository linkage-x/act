#!/usr/bin/env python3
"""
Inspect the first entry of an ACT HDF5 episode file and list all datasets.

Example:
    python inspect_hdf5_ee_action.py \
        --path /boot/common_data/2025/fr3_hdf5/1023_insert_tube_fr3_3dmouse_61ep_hdf5/episode_0.hdf5
"""

import argparse
import sys
from typing import Iterable

import h5py
import numpy as np


def format_array_sample(array: np.ndarray, max_elements: int = 16) -> str:
    """Create a concise string representation of a NumPy array sample."""
    flat = array.ravel()
    if flat.size <= max_elements:
        return np.array2string(flat, precision=4, suppress_small=True)
    preview = np.array2string(flat[:max_elements], precision=4, suppress_small=True)
    return f"{preview} ... (total {flat.size} elements)"


def iter_items(group: h5py.Group, prefix: str = "") -> Iterable[str]:
    """Recursively yield dataset metadata strings."""
    for key in sorted(group.keys()):
        item = group[key]
        path = f"{prefix}/{key}"
        if isinstance(item, h5py.Dataset):
            yield describe_dataset(item, path)
        elif isinstance(item, h5py.Group):
            yield f"{path}/ (group)"
            yield from iter_items(item, path)


def describe_dataset(dataset: h5py.Dataset, path: str) -> str:
    """Return a formatted string describing a dataset."""
    shape = dataset.shape
    dtype = dataset.dtype
    snippet = ""
    if shape and shape[0] > 0:
        try:
            sample = dataset[0]
            snippet = format_array_sample(np.asarray(sample))
        except Exception as exc:  # pragma: no cover - best effort only
            snippet = f"<unable to read sample: {exc}>"
    return f"{path}: shape={shape}, dtype={dtype}, sample[0]={snippet}"


def inspect_episode(path: str) -> None:
    """Open an episode HDF5 file and print dataset information."""
    with h5py.File(path, "r") as handle:
        attrs = dict(handle.attrs)
        print(f"Opened {path}")
        if attrs:
            print("File attributes:")
            for key, value in attrs.items():
                print(f"  {key}: {value}")
        else:
            print("File attributes: <none>")

        print("\nDatasets:")
        for line in iter_items(handle):
            print(f"  {line}")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ACT episode HDF5 structure.")
    parser.add_argument(
        "--path",
        required=True,
        help="Path to the episode_*.hdf5 file to inspect.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    inspect_episode(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

