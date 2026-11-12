"""Convert the robot rating dataset to ms-swift's messages format."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

LABEL_KEYS = ("closeness", "speed", "approach", "surprise")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_index_row(index_path: Path, line_number: int, line: str) -> Path:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{index_path}:{line_number}: invalid JSON: {error}"
        ) from error
    if not isinstance(row, dict):
        raise ValueError(f"{index_path}:{line_number}: each row must be a JSON object")

    sample_path = row.get("sample")
    if not isinstance(sample_path, str) or not sample_path:
        raise ValueError(f"{index_path}:{line_number}: sample must be a non-empty path")
    path = Path(sample_path)
    if not path.is_absolute():
        path = index_path.parent / path
    return path


def _read_sample(
    sample_path: Path, index_path: Path, line_number: int
) -> dict[str, Any]:
    try:
        with sample_path.open(encoding="utf-8") as sample_file:
            sample = json.load(sample_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"{index_path}:{line_number}: cannot read sample {sample_path}: {error}"
        ) from error
    if not isinstance(sample, dict):
        raise ValueError(
            f"{index_path}:{line_number}: sample must contain a JSON object"
        )
    return sample


def _rating(value: Any, key: str, sample_path: Path) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(f"{sample_path}: label {key!r} must be a number")
    try:
        rating = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"{sample_path}: label {key!r} must be a number") from error
    if not rating.is_finite() or not Decimal("1") <= rating <= Decimal("5"):
        raise ValueError(f"{sample_path}: label {key!r} must be between 1 and 5")
    if rating.normalize().as_tuple().exponent < -1:
        raise ValueError(
            f"{sample_path}: label {key!r} must have at most one decimal place"
        )
    return rating


def _record(
    sample: dict[str, Any],
    sample_path: Path,
    index_path: Path,
    line_number: int,
    asset_root: Path,
    sample_number: int,
) -> dict[str, Any]:
    frames = sample.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{sample_path}: frames must be a non-empty list")

    source_image_paths: list[Path] = []
    for frame in frames:
        if not isinstance(frame, str) or not frame:
            raise ValueError(
                f"{sample_path}: each frame path must be a non-empty string"
            )
        image_path = Path(frame)
        if not image_path.is_absolute():
            image_path = sample_path.parent / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise ValueError(
                f"{index_path}:{line_number}: image file does not exist: {image_path}"
            )
        source_image_paths.append(image_path)

    state = sample.get("state")
    if not isinstance(state, dict):
        raise ValueError(f"{sample_path}: state must be a JSON object")

    labels = sample.get("labels")
    if not isinstance(labels, dict):
        raise ValueError(f"{sample_path}: labels must be a JSON object")
    missing = [key for key in LABEL_KEYS if key not in labels]
    if missing:
        raise ValueError(f"{sample_path}: labels are missing {', '.join(missing)}")
    ratings = {key: _rating(labels[key], key, sample_path) for key in LABEL_KEYS}

    image_tokens = "\n".join("<image>" for _ in source_image_paths)
    label_keys = ", ".join(LABEL_KEYS)
    prompt = (
        "Rate the robots impression the pedestrians present in the camera frames using the robot state. "
        "The frames are ordered chronologically. Return only a JSON object with the "
        f"keys {label_keys}, each rated from 1.0 to 5.0 "
        "in increments of 0.1.\n\n"
        f"Camera frames:\n{image_tokens}\n\n"
        f"Robot state (JSON): {json.dumps(state, ensure_ascii=False)}"
    )
    pedestrians = sample.get("pedestrians")
    if pedestrians:
        prompt += f"\nTracked pedestrians (JSON): {json.dumps(pedestrians, ensure_ascii=False)}"

    sample_asset_dir = asset_root / f"sample_{sample_number:06d}"
    sample_asset_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[str] = []
    used_names: set[str] = set()
    for frame_number, source_image in enumerate(source_image_paths):
        image_name = source_image.name
        if image_name in used_names:
            image_name = f"frame_{frame_number:06d}_{image_name}"
        used_names.add(image_name)
        copied_image = sample_asset_dir / image_name
        shutil.copy2(source_image, copied_image)
        image_paths.append(
            Path(os.path.relpath(copied_image, start=PROJECT_ROOT)).as_posix()
        )

    response = (
        "{"
        + ", ".join(f'"{key}": {format(ratings[key], ".1f")}' for key in LABEL_KEYS)
        + "}"
    )
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "images": image_paths,
    }


def convert_data(input_path: str | Path, output_path: str | Path) -> Path:
    """Convert an index JSONL and its referenced samples to ms-swift JSONL."""
    source = Path(input_path).resolve()
    target = Path(output_path).resolve()
    asset_root = target.parent / f"{target.stem}_assets"
    records: list[dict[str, Any]] = []

    try:
        with source.open(encoding="utf-8") as index_file:
            for line_number, line in enumerate(index_file, start=1):
                if not line.strip():
                    continue
                sample_path = _read_index_row(source, line_number, line)
                sample = _read_sample(sample_path, source, line_number)
                records.append(
                    _record(
                        sample,
                        sample_path,
                        source,
                        line_number,
                        asset_root,
                        len(records),
                    )
                )
    except OSError as error:
        raise ValueError(f"cannot read input index {source}: {error}") from error

    if not records:
        raise ValueError(f"{source} contains no sample rows")

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert robot image sequences and ratings to ms-swift messages JSONL."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="data/train.jsonl",
        help="index JSONL containing sample paths (default: data/train.jsonl)",
    )
    parser.add_argument(
        "--out",
        default="msswift/smoke_data/train_converted.jsonl",
        help="output ms-swift JSONL path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = convert_data(args.input, args.out)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    print(f"Converted data written to {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
