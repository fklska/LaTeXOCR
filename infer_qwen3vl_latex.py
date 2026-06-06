"""Run LaTeX OCR inference with Qwen3-VL GGUF via llama.cpp.

This script intentionally shells out to llama.cpp's `llama-mtmd-cli`, because
Qwen3-VL GGUF support is freshest in llama.cpp itself and requires a separate
vision projector (`mmproj`) file.
"""

from __future__ import annotations

import argparse
import re
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ID = "fklska/Qwen3-VL-2B-Instruct-GGUF-LaTeX"
MODEL_FILENAME = "qwen3-vl-2b-instruct.Q4_K_M.gguf"
MMPROJ_FILENAME = "qwen3-vl-2b-instruct.F16-mmproj.gguf"
DEFAULT_MODEL_DIR = Path("models") / "qwen3vl-latex"
DEFAULT_PROMPT = (
    "Convert this image of a mathematical expression to LaTeX. "
    "Return only LaTeX."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Image-to-LaTeX inference for Qwen3-VL GGUF using llama.cpp."
    )
    parser.add_argument("--image", type=Path, help="Path to a PNG/JPG/WebP image.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Directory containing {MODEL_FILENAME} and {MMPROJ_FILENAME}.",
    )
    parser.add_argument(
        "--llama-bin",
        type=Path,
        default=None,
        help="Path to llama-mtmd-cli. If omitted, PATH and common local folders are searched.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt for the VL model.")
    parser.add_argument("--temp", type=float, default=0.1, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.8, help="Nucleus sampling value.")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling value.")
    parser.add_argument(
        "--max-tokens",
        "-n",
        type=int,
        default=1024,
        help="Maximum number of output tokens.",
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download the required GGUF files from Hugging Face, then exit unless --image is set.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the llama.cpp command without running inference.",
    )
    parser.add_argument(
        "--show-logs",
        action="store_true",
        help="Print llama.cpp stderr logs. By default, only the model answer is printed.",
    )
    return parser.parse_args()


def download_model_files(model_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for --download-model. Install it with:\n"
            "  pip install huggingface_hub"
        ) from exc

    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=[MODEL_FILENAME, MMPROJ_FILENAME],
        local_dir=str(model_dir),
    )


def resolve_llama_bin(explicit_path: Path | None) -> Path:
    executable_names = ["llama-mtmd-cli.exe", "llama-mtmd-cli"]

    if explicit_path is not None:
        candidate = explicit_path.expanduser()
        if candidate.is_file():
            return candidate
        raise SystemExit(f"llama.cpp binary was not found: {candidate}")

    for name in executable_names:
        found = shutil.which(name)
        if found:
            return Path(found)

    local_dirs = [
        Path("llama.cpp") / "build" / "bin" / "Release",
        Path("llama.cpp") / "build" / "bin",
        Path("bin"),
        Path("tools") / "llama.cpp",
    ]
    for directory in local_dirs:
        for name in executable_names:
            candidate = directory / name
            if candidate.is_file():
                return candidate

    raise SystemExit(
        "llama-mtmd-cli was not found.\n\n"
        "Install or download a recent llama.cpp build with Qwen3-VL support, then either:\n"
        "  1. add the directory containing llama-mtmd-cli to PATH, or\n"
        "  2. pass --llama-bin C:\\path\\to\\llama-mtmd-cli.exe\n\n"
        "The official llama.cpp releases are here:\n"
        "  https://github.com/ggml-org/llama.cpp/releases"
    )


def require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser()
    if not resolved.is_file():
        raise SystemExit(
            f"Missing {description}: {resolved}\n"
            f"Download the model files with:\n"
            f"  python infer_qwen3vl_latex.py --download-model"
        )
    return resolved


def build_command(args: argparse.Namespace, llama_bin: Path, model: Path, mmproj: Path) -> list[str]:
    image = require_file(args.image, "input image")
    return [
        str(llama_bin),
        "-m",
        str(model),
        "--mmproj",
        str(mmproj),
        "--image",
        str(image),
        "-p",
        args.prompt,
        "--temp",
        str(args.temp),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "-n",
        str(args.max_tokens),
    ]


def build_inference_command(
    image: Path,
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
    llama_bin: Path | None = None,
    prompt: str = DEFAULT_PROMPT,
    temp: float = 0.1,
    top_p: float = 0.8,
    top_k: int = 20,
    max_tokens: int = 1024,
) -> list[str]:
    model = require_file(model_dir / MODEL_FILENAME, "language model GGUF")
    mmproj = require_file(model_dir / MMPROJ_FILENAME, "vision projector GGUF")
    resolved_llama_bin = resolve_llama_bin(llama_bin)
    resolved_image = require_file(image, "input image")

    return [
        str(resolved_llama_bin),
        "-m",
        str(model),
        "--mmproj",
        str(mmproj),
        "--image",
        str(resolved_image),
        "-p",
        prompt,
        "--temp",
        str(temp),
        "--top-p",
        str(top_p),
        "--top-k",
        str(top_k),
        "-n",
        str(max_tokens),
    ]


def clean_model_output(output: str) -> str:
    text = output.strip()
    if not text:
        return ""

    fenced = re.search(r"```(?:latex|tex)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    prefixes = [
        "latex:",
        "LaTeX:",
        "Output:",
        "Answer:",
        "The LaTeX is:",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()

    return text.strip()


def infer_image_to_latex(
    image: Path,
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
    llama_bin: Path | None = None,
    prompt: str = DEFAULT_PROMPT,
    temp: float = 0.1,
    top_p: float = 0.8,
    top_k: int = 20,
    max_tokens: int = 1024,
    show_logs: bool = False,
) -> str:
    command = build_inference_command(
        image,
        model_dir=model_dir,
        llama_bin=llama_bin,
        prompt=prompt,
        temp=temp,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"llama.cpp inference failed with code {completed.returncode}: {details}")

    if completed.stderr and show_logs:
        print(completed.stderr.strip(), file=sys.stderr)
    return clean_model_output(completed.stdout)


def main() -> int:
    args = parse_args()

    if args.download_model:
        download_model_files(args.model_dir)
        print(f"Model files are ready in: {args.model_dir}")
        if args.image is None:
            return 0

    if args.image is None:
        raise SystemExit("--image is required unless only --download-model is used.")

    model = require_file(args.model_dir / MODEL_FILENAME, "language model GGUF")
    mmproj = require_file(args.model_dir / MMPROJ_FILENAME, "vision projector GGUF")
    llama_bin = resolve_llama_bin(args.llama_bin)
    command = build_command(args, llama_bin, model, mmproj)

    if args.dry_run:
        print(subprocess.list2cmdline(command) if os.name == "nt" else " ".join(command))
        return 0

    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.stdout:
        print(clean_model_output(completed.stdout))
    if completed.stderr and (args.show_logs or completed.returncode != 0):
        print(completed.stderr.strip(), file=sys.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
