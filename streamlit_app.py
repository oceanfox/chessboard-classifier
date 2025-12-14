"""
Interactive Streamlit demo for the chessboard classifier.

Launch with:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

import system
from fen_image_gen import (
    DEFAULT_TEMPLATE_FILE,
    load_templates as load_fen_templates,
    render_fen as render_fen_image,
    save_templates as save_fen_templates,
    train_templates as train_fen_templates,
)
from utils import utils

DATA_DIR = Path("data")
MODEL_FILES = {
    "Clean model": DATA_DIR / "model.clean.json.gz",
    "Noisy model": DATA_DIR / "model.noisy.json.gz",
}
IMAGE_ROOTS = {
    "Clean images": DATA_DIR / "clean",
    "Noisy images": DATA_DIR / "noisy",
}
BOARD_METADATA_FILE = DATA_DIR / "boards.dev.json"
PIECE_NAMES = {
    ".": "Empty",
    "P": "White pawn",
    "N": "White knight",
    "B": "White bishop",
    "R": "White rook",
    "Q": "White queen",
    "K": "White king",
    "p": "Black pawn",
    "n": "Black knight",
    "b": "Black bishop",
    "r": "Black rook",
    "q": "Black queen",
    "k": "Black king",
}
DEFAULT_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _focus_tab(label: str):
    """Inject JavaScript to focus the tab with the provided label."""
    escaped = json.dumps(label)
    components.html(
        f"""
        <script>
        const tabs = window.parent.document.querySelectorAll('button[role="tab"]');
        tabs.forEach((tab) => {{
            const text = tab.textContent || tab.innerText;
            if (text.trim() === {escaped}) {{
                tab.click();
            }}
        }});
        </script>
        """,
        height=0,
    )


@st.cache_resource
def load_model(model_key: str) -> dict:
    """Load and cache the trained model."""
    return utils.load_jsongz(str(MODEL_FILES[model_key]))


@st.cache_data
def load_board_metadata() -> List[dict]:
    """Load the development board metadata."""
    with open(BOARD_METADATA_FILE, "r", encoding="utf-8") as fp:
        return json.load(fp)


@st.cache_data
def load_feature_vectors(image_key: str, board_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Load feature vectors and labels for the first N dev boards."""
    metadata = load_board_metadata()
    subset = metadata[:board_count]
    images = utils.load_board_images(str(IMAGE_ROOTS[image_key]), subset)
    labels = utils.load_board_labels(subset)
    fvectors = system.images_to_feature_vectors(images)
    return fvectors, labels


def board_to_squares(image: Image.Image) -> List[np.ndarray]:
    """Split a PIL image into 64 grayscale square arrays."""
    gray = image.convert("L")
    board = np.array(gray)
    if board.shape[0] != board.shape[1]:
        raise ValueError("Board image must be square (equal width and height).")
    square_size = board.shape[0] // 8
    if square_size * 8 != board.shape[0]:
        raise ValueError("Board size must be divisible by 8.")

    squares = []
    for row in range(8):
        for col in range(8):
            r0, r1 = row * square_size, (row + 1) * square_size
            c0, c1 = col * square_size, (col + 1) * square_size
            squares.append(board[r0:r1, c0:c1])
    return squares


def classify_board(image: Image.Image, model: dict, return_intermediate: bool = False):
    """Run the PCA+LDA+weighted-KNN pipeline on a board image."""
    squares = board_to_squares(image)
    fvectors = system.images_to_feature_vectors(squares)
    reduced = system.reduce_dimensions(fvectors, model)
    predictions = system.classify_boards(reduced, model)
    if return_intermediate:
        return predictions[:64], squares, fvectors, reduced
    return predictions[:64]


def draw_board(labels: List[str], title: str, highlight_mask: Optional[List[bool]] = None):
    """Render a chessboard heatmap using matplotlib."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    for row in range(8):
        for col in range(8):
            idx = row * 8 + col
            default_color = "#f0d9b5" if (row + col) % 2 == 0 else "#b58863"
            fill = default_color
            if highlight_mask and highlight_mask[idx]:
                fill = "#f28b82"
            rect = patches.Rectangle(
                (col, 7 - row),
                1,
                1,
                linewidth=0.5,
                edgecolor="black",
                facecolor=fill,
            )
            ax.add_patch(rect)

            label = labels[idx]
            display_label = "" if label == "." else label
            ax.text(
                col + 0.5,
                7 - row + 0.5,
                display_label,
                ha="center",
                va="center",
                fontsize=14,
                color="black",
            )

    return fig


def flatten_board(board_rows: List[str]) -> List[str]:
    """Convert board rows from metadata to a flat list of 64 labels."""
    return list("".join(board_rows))


def compute_embeddings(fvectors: np.ndarray, model: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return PCA and LDA projections for visualization."""
    pca_vectors = np.array(model["pca_eigenvectors"])
    lda_vectors = np.array(model["lda_eigenvectors"])

    centered = fvectors - np.mean(fvectors, axis=0)
    pca_data = system.run_pca(centered, pca_vectors)
    lda_data = system.run_lda((pca_data - np.mean(pca_data, axis=0)), lda_vectors)
    return pca_data, lda_data

@st.cache_resource
def load_template_file_cached(template_path: str):
    return load_fen_templates(Path(template_path))


def main():
    st.set_page_config(page_title="Chessboard Classifier Demo", layout="wide")
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            width: 270px !important;
        }
        section[data-testid="stSidebar"] > div {
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Chessboard Classifier")
    st.write(
        "Interactively run the PCA + LDA + weighted KNN pipeline from the COM2004 assignment "
        "on clean or noisy chessboard scans. Choose any board from the development set and "
        "compare model predictions across noise settings. Use the PCA & LDA explorer tab to "
        "see how squares cluster in the reduced feature spaces."
    )

    metadata = load_board_metadata()
    st.sidebar.header("Configuration")
    model_options = list(MODEL_FILES.keys())
    model_choice = st.sidebar.segmented_control(
        "Select model",
        model_options,
        default=st.session_state.get("model_choice", model_options[0]),
    )
    st.session_state["model_choice"] = model_choice
    model = load_model(model_choice)

    tab_labels = ["Classifier", "PCA & LDA Explorer", "FEN Generator"]
    tab_classifier, tab_projection, tab_fen = st.tabs(tab_labels)

    with tab_classifier:
        board_image: Optional[Image.Image] = None
        true_layout: Optional[List[str]] = None

        selected_index = st.sidebar.slider(
            "Board index", min_value=1, max_value=len(metadata), value=1
        )
        selected_board = metadata[selected_index - 1]
        image_options = list(IMAGE_ROOTS.keys())
        image_set = st.sidebar.segmented_control(
            "Image quality",
            image_options,
            default=st.session_state.get("image_set", image_options[0]),
        )
        st.session_state["image_set"] = image_set
        image_path = IMAGE_ROOTS[image_set] / selected_board["image"]
        board_image = Image.open(image_path)
        true_layout = flatten_board(selected_board["board"])
        st.sidebar.markdown(f"**FEN**: `{selected_board['fen']}`")

        if board_image is None:
            st.stop()

        try:
            predictions, _, debug_fvectors, debug_reduced = classify_board(
                board_image, model, return_intermediate=True
            )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        truth_board = true_layout or ["?"] * 64
        mismatch_mask = (
            [pred != truth for pred, truth in zip(predictions, truth_board)]
            if true_layout
            else None
        )

        st.subheader("Results")
        if true_layout:
            matches = sum(pred == truth for pred, truth in zip(predictions, true_layout))
            st.write(
                f"Square accuracy: **{matches / 64 * 100:.1f}%** "
                f"({matches}/64 correct). "
                f"Board prediction: **{'correct' if matches == 64 else 'incorrect'}**."
            )
        else:
            st.write(
                "Predicted board shown below. Uploading a labeled board will display accuracy stats."
            )

        if true_layout:
            cols = st.columns(3)
        else:
            cols = st.columns(2)

        with cols[0]:
            st.subheader("Input board")
            st.image(board_image, caption="Input board", width="stretch")

        with cols[1]:
            st.subheader("Predicted layout")
            fig = draw_board(predictions, f"Model: {model_choice}", mismatch_mask)
            st.pyplot(fig)

        if true_layout:
            with cols[2]:
                st.subheader("True layout")
                fig_truth = draw_board(true_layout, "Expected board state")
                st.pyplot(fig_truth)

        debug_details = {
            "board_index": selected_index,
            "image_set": image_set,
            "image_size": getattr(board_image, "size", None),
            "image_mode": getattr(board_image, "mode", None),
            "fvectors_shape": list(debug_fvectors.shape),
            "reduced_shape": list(debug_reduced.shape),
            "prediction_sample": predictions[:8],
            "truth_sample": truth_board[:8],
        }
        if true_layout:
            debug_details["mismatch_indices"] = [
                idx for idx, (pred, truth) in enumerate(zip(predictions, truth_board)) if pred != truth
            ]
        with st.expander("Debug data (classifier)", expanded=False):
            st.json(debug_details)

        st.caption(
            "Tip: toggle between clean/noisy models and images to explore how robustness changes."
        )

    with tab_projection:
        st.subheader("PCA & LDA Visualisation")
        st.write(
            "Each point represents a square from the selected development boards, coloured by its true label. "
            "Use this explorer to see how PCA and LDA separate the classes."
        )

        col_config, col_info = st.columns([1, 1])

        with col_config:
            data_options = list(IMAGE_ROOTS.keys())
            dataset_choice = st.segmented_control(
                "Image set",
                data_options,
                default=st.session_state.get("dataset_choice", data_options[0]),
            )
            st.session_state["dataset_choice"] = dataset_choice
            board_count = st.slider(
                "Number of boards",
                min_value=1,
                max_value=len(metadata),
                value=min(5, len(metadata)),
            )

        fvectors, labels = load_feature_vectors(dataset_choice, board_count)
        pca_data, lda_data = compute_embeddings(fvectors, model)
        human_labels = [PIECE_NAMES.get(lbl, lbl) for lbl in labels]

        st.write(f"Visualising {len(labels)} squares ({board_count} boards).")

        fig_pca = px.scatter_3d(
            x=pca_data[:, 0],
            y=pca_data[:, 1],
            z=pca_data[:, 2],
            color=human_labels,
            labels={"x": "PC1", "y": "PC2", "z": "PC3", "color": "Label"},
            hover_data={"Square label": labels},
            title="First three PCA components",
        )
        st.plotly_chart(fig_pca, width="stretch")

        fig_lda = px.scatter(
            x=lda_data[:, 0],
            y=lda_data[:, 1],
            color=human_labels,
            labels={"x": "LDA1", "y": "LDA2", "color": "Label"},
            hover_data={"Square label": labels},
            title="First two LDA components",
        )
        st.plotly_chart(fig_lda, width="stretch")

        projection_debug = {
            "dataset_choice": dataset_choice,
            "board_count": board_count,
            "fvectors_shape": list(fvectors.shape),
            "pca_shape": list(pca_data.shape),
            "lda_shape": list(lda_data.shape),
            "label_counts": {label: int(count) for label, count in zip(*np.unique(labels, return_counts=True))},
        }
        with st.expander("Debug data (PCA/LDA)", expanded=False):
            st.json(projection_debug)

        with col_info:
            st.markdown(
                "**How to read this:**\n"
                "- PCA highlights raw pixel variance; pawns often form tight planes due to similar shapes.\n"
                "- LDA enforces class separation, so better separated clusters imply easier classification.\n"
            )

    with tab_fen:
        st.subheader("FEN Image Generator")
        st.write(
            "Generate a synthetic 400×400 board rendered in the same style as the dataset by combining "
            "average tiles for each (piece, square colour) pair. Provide a FEN placement string and a "
            "template library created via the `fen_image_tool.py` script (or train one below)."
        )

        fen_input = st.text_input("FEN string", value=DEFAULT_FEN)

        with st.expander("More customization", expanded=False):
            template_path_input = st.text_input(
                "Template file (.npz)",
                value=str(DEFAULT_TEMPLATE_FILE),
                help="Path to a template library file.",
            )

            col_meta, col_root = st.columns(2)
            metadata_path_input = col_meta.text_input(
                "Metadata JSON for training",
                value=str(DATA_DIR / "boards.train.json"),
                help="Used if you retrain the template library.",
            )
            image_root_input = col_root.text_input(
                "Image root directory",
                value=str(DATA_DIR / "clean"),
                help="Directory containing the board images referenced by the metadata file.",
            )
            train_clicked = st.button("Train/refresh template library")

        if train_clicked:
            st.session_state["switch_tab_label"] = "FEN Generator"
            metadata_path = Path(metadata_path_input)
            image_root = Path(image_root_input)
            output_path = Path(template_path_input)
            try:
                with st.spinner("Training templates..."):
                    templates, square_shape = train_fen_templates(metadata_path, image_root)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    save_fen_templates(output_path, templates, square_shape)
                    load_template_file_cached.clear()
                st.success(f"Saved {len(templates)} templates to {output_path}")
            except FileNotFoundError as exc:
                st.error(f"File not found: {exc}")
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Failed to train templates: {exc}")

        generate_clicked = st.button("Generate board image", type="primary")
        if generate_clicked:
            st.session_state["switch_tab_label"] = "FEN Generator"
            template_path = Path(template_path_input)
            try:
                templates, square_shape = load_template_file_cached(str(template_path))
            except FileNotFoundError:
                st.error(f"Template file not found: {template_path}")
                templates = square_shape = None
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Failed to load templates: {exc}")
                templates = square_shape = None

            if templates is not None and square_shape is not None:
                try:
                    image = render_fen_image(fen_input, templates, square_shape)
                    if image.size != (400, 400):
                        image = image.resize((400, 400), Image.BICUBIC)
                except ValueError as exc:
                    st.error(f"Invalid FEN string: {exc}")
                except Exception as exc:  # pylint: disable=broad-except
                    st.error(f"Failed to render board: {exc}")
                else:
                    st.image(image, caption="Generated board", width=400)
                    buffer = BytesIO()
                    image.save(buffer, format="PNG")
                    buffer.seek(0)
                    st.download_button(
                        label="Download PNG",
                        data=buffer,
                        file_name="generated_board.png",
                        mime="image/png",
                    )

    pending_tab = st.session_state.pop("switch_tab_label", None)
    if pending_tab:
        _focus_tab(pending_tab)


if __name__ == "__main__":
    main()
