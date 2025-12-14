"""Generate 400x400 chessboard images from FEN strings using dataset style.

This script builds average square templates from the provided training boards
and then reassembles them to synthesize new boards that follow the same style
as the COM2004 assignment images.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from PIL import Image

from utils import utils

SQUARES_PER_SIDE = 8
DEFAULT_TEMPLATE_FILE = Path("data") / "piece_templates.clean.npz"


def _square_color(row: int, col: int) -> str:
    """Return the board color ('white' or 'black') for the (row, col) square."""
    return "white" if (row + col) % 2 == 0 else "black"


def _flatten_board(board_rows: Iterable[str]) -> list[str]:
    """Flatten a list of 8 row strings into a single list of 64 characters."""
    flat: list[str] = []

    for row in board_rows:
        flat.extend(list(row))

    if len(flat) != SQUARES_PER_SIDE ** 2:
        raise ValueError("Board metadata must have exactly 64 squares.")
    return flat


def _parse_fen(fen: str) -> list[list[str]]:
    """Convert the board portion of a FEN string into an 8x8 array."""
    board_part = fen.split()[0]
    ranks = board_part.split("/")
    if len(ranks) != SQUARES_PER_SIDE:
        raise ValueError("FEN must contain 8 ranks.")

    board: list[list[str]] = []
    for rank in ranks:
        expanded: list[str] = []
        for piece in rank:
            if piece.isdigit():
                expanded.extend("." * int(piece))
            else:
                expanded.append(piece)
        if len(expanded) != SQUARES_PER_SIDE:
            raise ValueError(f"Invalid rank '{rank}' in FEN.")
        board.append(expanded)
    return board


def train_templates(metadata_path: Path, image_root: Path) -> \
    Tuple[Dict[str, np.ndarray], Tuple[int, ...]]:
    """Create an average square template for each (label, color) pair."""
    with open(metadata_path, "r", encoding="utf-8") as fp:
        metadata = json.load(fp)

    template_sums: Dict[str, np.ndarray] = {}
    template_counts: Dict[str, int] = {}
    square_shape: Tuple[int, ...] | None = None

    for entry in metadata:
        image_path = image_root / entry["image"]
        squares = utils.load_square_images(str(image_path))
        if square_shape is None:
            square_shape = squares[0].shape
        labels = _flatten_board(entry["board"])

        for idx, (square, label) in enumerate(zip(squares, labels)):
            row, col = divmod(idx, SQUARES_PER_SIDE)
            key = f"{label}|{_square_color(row, col)}"
            square_f = square.astype(np.float32)

            template_sums.setdefault(key, np.zeros_like(square_f, dtype=np.float32))
            template_sums[key] += square_f
            template_counts[key] = template_counts.get(key, 0) + 1

    templates: Dict[str, np.ndarray] = {}
    for key, total in template_sums.items():
        templates[key] = total / template_counts[key]

    if square_shape is None:
        raise ValueError("No board images found while training templates.")

    return templates, square_shape


def save_templates(npz_path: Path, templates: Dict[str, np.ndarray], square_shape: Tuple[int, ...]) -> None:
    """Persist template arrays to a compressed .npz file."""
    arrays = {key: value for key, value in templates.items()}
    arrays["_square_shape"] = np.array(square_shape, dtype=np.int32)
    np.savez_compressed(npz_path, **arrays)


def load_templates(npz_path: Path) -> Tuple[Dict[str, np.ndarray], Tuple[int, ...]]:
    """Load template arrays from a compressed .npz file."""
    data = np.load(npz_path)
    square_shape = tuple(int(dim) for dim in data["_square_shape"])
    templates = {key: data[key] for key in data.files if key != "_square_shape"}
    return templates, square_shape


def _empty_fallback_key(templates: Dict[str, np.ndarray], color: str) -> str | None:
    """Return the template key for an empty square fallback of the desired color."""
    key = f".|{color}"
    if key in templates:
        return key
    alternate = f".|{'black' if color == 'white' else 'white'}"
    if alternate in templates:
        return alternate
    return None


def render_fen(
    fen: str,
    templates: Dict[str, np.ndarray],
    square_shape: Tuple[int, ...],
) -> Image.Image:
    """Generate a PIL image for the provided FEN using the learned templates."""
    board = _parse_fen(fen)
    tile_h, tile_w = square_shape[:2]
    board_shape = (tile_h * SQUARES_PER_SIDE, tile_w * SQUARES_PER_SIDE) + square_shape[2:]
    board_img = np.zeros(board_shape, dtype=np.float32)

    for row in range(SQUARES_PER_SIDE):
        for col in range(SQUARES_PER_SIDE):
            label = board[row][col]
            color = _square_color(row, col)

            key = f"{label}|{color}"
            tile = templates.get(key)
            if tile is None:
                fallback_color = "black" if color == "white" else "white"
                tile = templates.get(f"{label}|{fallback_color}")
            if tile is None:
                empty_key = _empty_fallback_key(templates, color)
                if empty_key is None:
                    raise RuntimeError("No fallback template available for empty squares.")
                tile = templates[empty_key]

            r0, r1 = row * tile_h, (row + 1) * tile_h
            c0, c1 = col * tile_w, (col + 1) * tile_w
            board_img[r0:r1, c0:c1, ...] = tile

    board_uint8 = np.clip(board_img, 0, 255).astype(np.uint8)
    mode = "L" if board_uint8.ndim == 2 else "RGB"
    return Image.fromarray(board_uint8, mode=mode)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate dataset-style boards from FEN strings.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Create piece templates from board metadata.")
    train_parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data") / "boards.train.json",
        help="Path to the JSON file describing the training boards.",
    )
    train_parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("data") / "clean",
        help="Root directory containing the board images referenced in the metadata file.",
    )
    train_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_TEMPLATE_FILE,
        help="Where to write the generated template library (.npz).",
    )

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--fen", required=True, help="FEN string (only placement section needed).")
    render_parser.add_argument(
        "--templates",
        type=Path,
        default=DEFAULT_TEMPLATE_FILE,
        help="Template .npz (produced by train command)",
    )
    render_parser.add_argument(
        "--output",
        type=Path,
        default=Path("generated.png"),
    )

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.command == "train":
        templates, square_shape = train_templates(args.metadata, args.image_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)

        save_templates(args.output, templates, square_shape)
        print(f"Saved {len(templates)} templates to {args.output}")
    elif args.command == "render":
        # Load templates
        templates, square_shape = load_templates(args.templates)

        # Render the image
        image = render_fen(args.fen, templates, square_shape)
        image = image.resize((400, 400), Image.BICUBIC) if image.width != 400 else image

        # Save output image
        image.save(args.output)
        print(f"Wrote synthesized board to {args.output}")


if __name__ == "__main__":
    main()
