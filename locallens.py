#!/usr/bin/env python3
"""
locallens.py — The Wanderer Chronicles Episode 1
Generates a vertical MP4 video using Stable Diffusion + OpenCV + pyttsx3 + FFmpeg
"""

import argparse
import os
import sys
import platform
import subprocess
import tempfile
import shutil
import time
import textwrap
from pathlib import Path

# ── third-party imports (checked at runtime) ──────────────────────────────────
try:
    import torch
except ImportError:
    sys.exit("[ERROR] torch not installed. Run: pip install torch")

try:
    import numpy as np
except ImportError:
    sys.exit("[ERROR] numpy not installed. Run: pip install numpy")

try:
    import cv2
except ImportError:
    sys.exit("[ERROR] opencv-python not installed. Run: pip install opencv-python")

try:
    from PIL import Image
except ImportError:
    sys.exit("[ERROR] Pillow not installed. Run: pip install Pillow")

try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
except ImportError:
    sys.exit("[ERROR] diffusers not installed. Run: pip install diffusers transformers accelerate")

try:
    import pyttsx3
except ImportError:
    sys.exit("[ERROR] pyttsx3 not installed. Run: pip install pyttsx3")

try:
    import ffmpeg
except ImportError:
    sys.exit("[ERROR] ffmpeg-python not installed. Run: pip install ffmpeg-python")


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

SERIES_TITLE   = "The Wanderer Chronicles"
EPISODE_NUMBER = 1
OUTPUT_FILENAME = f"locallens_episode_{EPISODE_NUMBER}.mp4"

VIDEO_WIDTH    = 1080
VIDEO_HEIGHT   = 1920
FPS            = 30
SECONDS_PER_IMAGE = 3
FRAMES_PER_CLIP   = FPS * SECONDS_PER_IMAGE   # 90 frames

SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"

VOICEOVER_SCRIPT = (
    "In the highlands where ancient rivers meet the sky, "
    "they move like living poetry — Habesha youth, draped in the warmth "
    "of their heritage. Their eyes hold the depth of Ethiopia's timeless story, "
    "proud and luminous, written long before words existed."
)

# Episode 1 — "Roots & Restless Winds" scene prompts
EPISODE_SCENE_PROMPTS = [
    (
        "Portrait of young Habesha Ethiopian teenagers, boy and girl, "
        "wearing traditional white habesha kemis with colorful embroidery and netela shawl, "
        "standing in sun-drenched Addis Ababa highlands, golden hour light, "
        "warm brown skin tones, natural features, proud expressions, "
        "cinematic photography, 8k, photorealistic, vibrant colors, "
        "Ethiopian cultural heritage, beautiful faces, NOT dark-skinned caricature"
    ),
    (
        "Vibrant Ethiopian market scene, young Habesha boy and girl browsing "
        "colorful spices and woven textiles, traditional habesha clothing, "
        "Addis Ababa cobblestone streets, warm afternoon sunlight, "
        "ancient map fragment visible in girl's hand, curious expressions, "
        "cinematic wide shot, photorealistic, rich cultural details, "
        "Ethiopian youth, natural skin tones, beautiful authentic faces"
    ),
    (
        "Two Ethiopian teenagers sitting on a rooftop at golden dusk, "
        "Addis Ababa skyline behind them, boy holding an acceptance letter, "
        "girl examining half of an old hand-drawn map, "
        "traditional habesha kemis and modern casual clothes mix, "
        "warm amber and purple sunset sky, intimate conversation, "
        "cinematic portrait photography, photorealistic, 8k resolution, "
        "Habesha youth, natural features, expressive eyes"
    ),
    (
        "Close-up detail of intricate habesha kemis embroidery with hidden map fragments "
        "woven into the fabric pattern, Ethiopian traditional textile art, "
        "gold and white threads, ancient symbols, warm candlelight, "
        "macro photography, photorealistic, rich textures, "
        "cultural heritage artifact, mysterious and beautiful, "
        "Ethiopian craftsmanship, cinematic lighting"
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def print_banner():
    """Print a styled startup banner."""
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║           L O C A L L E N S  —  Video Generator             ║
║   {SERIES_TITLE:<56}║
║   Episode {EPISODE_NUMBER:<51}║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def progress(step: int, total: int, label: str = ""):
    """Simple ASCII progress bar."""
    bar_len = 40
    filled  = int(bar_len * step / total)
    bar     = "█" * filled + "░" * (bar_len - filled)
    pct     = int(100 * step / total)
    print(f"\r  [{bar}] {pct:3d}%  {label}", end="", flush=True)
    if step == total:
        print()


def get_desktop_path() -> Path:
    """Return the user's Desktop path cross-platform."""
    system = platform.system()
    if system == "Windows":
        desktop = Path(os.environ.get("USERPROFILE", Path.home())) / "Desktop"
    elif system == "Darwin":
        desktop = Path.home() / "Desktop"
    else:  # Linux / other
        # Respect XDG if set
        xdg = os.environ.get("XDG_DESKTOP_DIR")
        desktop = Path(xdg) if xdg else Path.home() / "Desktop"

    desktop.mkdir(parents=True, exist_ok=True)
    return desktop


def detect_device() -> torch.device:
    """Detect best available compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name   = torch.cuda.get_device_name(0)
        vram   = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✔ GPU detected : {name}  ({vram:.1f} GB VRAM)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  ✔ Apple MPS (Metal) detected")
    else:
        device = torch.device("cpu")
        print("  ⚠ No GPU found — falling back to CPU (slower)")
    return device


def check_ffmpeg_binary():
    """Ensure the ffmpeg binary is on PATH."""
    if shutil.which("ffmpeg") is None:
        sys.exit(
            "[ERROR] ffmpeg binary not found on PATH.\n"
            "  Windows : winget install ffmpeg  OR  choco install ffmpeg\n"
            "  Mac     : brew install ffmpeg\n"
            "  Linux   : sudo apt install ffmpeg"
        )
    print("  ✔ ffmpeg binary found")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — STABLE DIFFUSION IMAGE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def load_sd_pipeline(device: torch.device) -> StableDiffusionPipeline:
    """Download / load the Stable Diffusion pipeline."""
    print(f"\n[1/4] Loading Stable Diffusion model: {SD_MODEL_ID}")
    print("      (First run downloads ~4 GB — subsequent runs use cache)\n")

    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        SD_MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,          # disable for speed; re-enable for production
        requires_safety_checker=False,
    )

    # Faster scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # Memory optimisations
    if device.type == "cuda":
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("  ✔ xformers memory-efficient attention enabled")
        except Exception:
            pass  # xformers optional

    print("  ✔ Pipeline loaded\n")
    return pipe


def generate_images(
    pipe: StableDiffusionPipeline,
    prompts: list[str],
    work_dir: Path,
    device: torch.device,
) -> list[Path]:
    """Generate one 768×768 image per prompt, then resize to 1080×1920."""
    print("[2/4] Generating images with Stable Diffusion …\n")

    negative_prompt = (
        "blurry, low quality, distorted face, extra limbs, deformed, "
        "cartoon, anime, sketch, dark skin caricature, racist depiction, "
        "ugly, bad anatomy, watermark, text, logo"
    )

    image_paths: list[Path] = []

    for idx, prompt in enumerate(prompts, start=1):
        print(f"  Scene {idx}/{len(prompts)}: generating …")
        t0 = time.time()

        # Use 512×512 on CPU to keep it tractable; 768×768 on GPU
        size = 512 if device.type == "cpu" else 768

        result = pipe(
            prompt          = prompt,
            negative_prompt = negative_prompt,
            width           = size,
            height          = size,
            num_inference_steps = 25 if device.type == "cpu" else 40,
            guidance_scale  = 7.5,
            num_images_per_prompt = 1,
        )

        pil_img: Image.Image = result.images[0]

        # ── Resize / crop to 1080×1920 (vertical) ──────────────────────────
        pil_img = resize_and_crop(pil_img, VIDEO_WIDTH, VIDEO_HEIGHT)

        out_path = work_dir / f"scene_{idx:02d}.png"
        pil_img.save(out_path, "PNG")
        image_paths.append(out_path)

        elapsed = time.time() - t0
        print(f"    ✔ Saved {out_path.name}  ({elapsed:.1f}s)")
        progress(idx, len(prompts), f"Scene {idx} done")

    print()
    return image_paths


def resize_and_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """
    Scale image so it fills target dimensions, then centre-crop.
    Preserves aspect ratio without black bars.
    """
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    img   = img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - target_w) // 2
    top  = (new_h - target_h) // 2
    img  = img.crop((left, top, left + target_w, top + target_h))
    return img


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — OPENCV ANIMATION (zoom / pan)
# ══════════════════════════════════════════════════════════════════════════════

def apply_ken_burns(
    img_path: Path,
    out_path: Path,
    effect: str = "zoom_in",
) -> Path:
    """
    Render a Ken Burns (zoom/pan) animated clip from a single image.

    Parameters
    ----------
    img_path : source PNG
    out_path : destination AVI (lossless, for FFmpeg assembly)
    effect   : one of 'zoom_in', 'zoom_out', 'pan_left', 'pan_right'
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    h, w = img_bgr.shape[:2]  # should be 1920 × 1080

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (w, h))

    ZOOM_START = 1.00
    ZOOM_END   = 1.15   # 15 % zoom over 3 s
    PAN_PIXELS = 80     # pixels to pan over 3 s

    for frame_idx in range(FRAMES_PER_CLIP):
        t = frame_idx / (FRAMES_PER_CLIP - 1)   # 0.0 → 1.0
        t_ease = ease_in_out(t)

        if effect == "zoom_in":
            scale  = ZOOM_START + (ZOOM_END - ZOOM_START) * t_ease
            frame  = zoom_frame(img_bgr, scale)

        elif effect == "zoom_out":
            scale  = ZOOM_END - (ZOOM_END - ZOOM_START) * t_ease
            frame  = zoom_frame(img_bgr, scale)

        elif effect == "pan_left":
            offset_x = int(PAN_PIXELS * t_ease)
            frame    = pan_frame(img_bgr, offset_x, 0)

        elif effect == "pan_right":
            offset_x = int(-PAN_PIXELS * t_ease)
            frame    = pan_frame(img_bgr, offset_x, 0)

        else:
            frame = img_bgr.copy()

        writer.write(frame)

    writer.release()
    return out_path


def ease_in_out(t: float) -> float:
    """Smooth cubic ease-in-out interpolation."""
    return t * t * (3.0 - 2.0 * t)


def zoom_frame(img: np.ndarray, scale: float) -> np.ndarray:
    """Return a centre-zoomed crop of `img` scaled back to original size."""
    h, w = img.shape[:2]
    new_w = int(w / scale)
    new_h = int(h / scale)
    x1    = (w - new_w) // 2
    y1    = (h - new_h) // 2
    crop  = img[y1:y1 + new_h, x1:x1 + new_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def pan_frame(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Return a translated crop of `img` (clamped to borders)."""
    h, w = img.shape[:2]
    x1   = max(0, min(dx, w - 1))
    y1   = max(0, min(dy, h - 1))
    x2   = x1 + w
    y2   = y1 + h

    # Pad image so we can always crop w×h
    padded = cv2.copyMakeBorder(
        img,
        top    = abs(min(dy, 0)),
        bottom = max(dy, 0),
        left   = abs(min(dx, 0)),
        right  = max(dx, 0),
        borderType = cv2.BORDER_REFLECT_101,
    )
    ph, pw = padded.shape[:2]
    cx1 = (pw - w) // 2 + dx
    cy1 = (ph - h) // 2 + dy
    cx1 = max(0, min(cx1, pw - w))
    cy1 = max(0, min(cy1, ph - h))
    return padded[cy1:cy1 + h, cx1:cx1 + w]


def create_animated_clips(image_paths: list[Path], work_dir: Path) -> list[Path]:
    """Apply Ken Burns effects to all images and return clip paths."""
    print("[3/4] Creating animated clips …\n")

    effects = ["zoom_in", "pan_left", "zoom_out", "pan_right"]
    clip_paths: list[Path] = []

    for idx, img_path in enumerate(image_paths):
        effect    = effects[idx % len(effects)]
        clip_path = work_dir / f"clip_{idx + 1:02d}.avi"
        print(f"  Clip {idx + 1}/{len(image_paths)}: {effect} → {clip_path.name}")
        apply_ken_burns(img_path, clip_path, effect=effect)
        clip_paths.append(clip_path)
        progress(idx + 1, len(image_paths), f"Clip {idx + 1} done")

    print()
    return clip_paths


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — VOICEOVER GENERATION (pyttsx3)
# ══════════════════════════════════════════════════════════════════════════════

def generate_voiceover(script: str, work_dir: Path) -> Path:
    """
    Synthesise speech from `script` and save as WAV.
    Returns path to the WAV file.
    """
    print("[4/4] Generating voiceover …")
    audio_path = work_dir / "voiceover.wav"

    engine = pyttsx3.init()

    # ── Voice selection ──────────────────────────────────────────────────────
    voices = engine.getProperty("voices")
    # Prefer a female voice for warmth; fall back to first available
    chosen = None
    for v in voices:
        if "female" in v.name.lower() or "zira" in v.name.lower() or "samantha" in v.name.lower():
            chosen = v.id
            break
    if chosen:
        engine.setProperty("voice", chosen)
        print(f"  ✔ Voice: {chosen}")
    else:
        print(f"  ✔ Voice: {voices[0].name if voices else 'default'}")

    engine.setProperty("rate",   145)   # words per minute
    engine.setProperty("volume", 1.0)

    engine.save_to_file(script, str(audio_path))
    engine.runAndWait()

    if not audio_path.exists() or audio_path.stat().st_size < 1000:
        raise RuntimeError(
            "pyttsx3 failed to write voiceover.wav — "
            "check your system TTS engine is installed."
        )

    print(f"  ✔ Voiceover saved: {audio_path.name}\n")
    return audio_path


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — FFMPEG VIDEO ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def assemble_video(
    clip_paths: list[Path],
    audio_path: Path,
    output_path: Path,
    work_dir: Path,
) -> None:
    """
    Concatenate animated clips, mix in voiceover, and export final MP4.
    Uses subprocess + FFmpeg concat demuxer for reliability.
    """
    print("[5/5] Assembling final video with FFmpeg …\n")

    # ── 1. Write concat list ─────────────────────────────────────────────────
    concat_list = work_dir / "concat.txt"
    with open(concat_list, "w") as f:
        for cp in clip_paths:
            # FFmpeg requires forward slashes even on Windows
            f.write(f"file '{cp.as_posix()}'\n")

    # ── 2. Concatenate clips → silent_video.mp4 ──────────────────────────────
    silent_video = work_dir / "silent_video.mp4"
    concat_cmd = [
        "ffmpeg", "-y",
        "-f",  "concat",
        "-safe", "0",
        "-i",  str(concat_list),
        "-vf", f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=decrease,"
               f"pad={VIDEO_WIDTH}:{VIDEO_HEIGHT}:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf",    "18",
        "-pix_fmt", "yuv420p",
        str(silent_video),
    ]
    _run_ffmpeg(concat_cmd, "Concatenating clips")

    # ── 3. Get durations ─────────────────────────────────────────────────────
    video_duration = len(clip_paths) * SECONDS_PER_IMAGE   # seconds
    audio_duration = _probe_duration(audio_path)

    print(f"  Video duration : {video_duration}s")
    print(f"  Audio duration : {audio_duration:.1f}s")

    # ── 4. Pad or trim audio to match video ──────────────────────────────────
    padded_audio = work_dir / "audio_padded.wav"
    if audio_duration < video_duration:
        # Pad with silence
        pad_cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-af", f"apad=whole_dur={video_duration}",
            "-t",  str(video_duration),
            str(padded_audio),
        ]
    else:
        # Trim to video length
        pad_cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-t", str(video_duration),
            str(padded_audio),
        ]
    _run_ffmpeg(pad_cmd, "Adjusting audio length")

    # ── 5. Merge video + audio → final MP4 ───────────────────────────────────
    merge_cmd = [
        "ffmpeg", "-y",
        "-i", str(silent_video),
        "-i", str(padded_audio),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path),
    ]
    _run_ffmpeg(merge_cmd, "Merging video + audio")

    size_mb = output_path.stat().st_size / 1e6
    print(f"\n  ✔ Final video : {output_path}")
    print(f"  ✔ File size   : {size_mb:.1f} MB\n")


def _run_ffmpeg(cmd: list[str], label: str) -> None:
    """Run an FFmpeg command, streaming stderr for progress."""
    print(f"  → {label} …")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        print(f"\n[FFmpeg ERROR] {label} failed:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"FFmpeg step failed: {label}")
    print(f"    ✔ {label} complete")


def _probe_duration(path: Path) -> float:
    """Return media duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return float(SECONDS_PER_IMAGE * 4)   # safe fallback


# ══════════════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog        = "locallens.py",
        description = (
            f"LocalLens — AI Video Generator\n"
            f"Series : {SERIES_TITLE}\n"
            f"Episode: {EPISODE_NUMBER}"
        ),
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = textwrap.dedent("""
            Examples:
              python locallens.py
              python locallens.py --prompt "habesha youth in Addis Ababa highlands"
              python locallens.py --prompt "your custom prompt" --scenes 4
              python locallens.py --cpu   (force CPU even if GPU available)
        """),
    )
    parser.add_argument(
        "--prompt",
        type    = str,
        default = None,
        help    = "Override the default video prompt (appended to each scene prompt)",
    )
    parser.add_argument(
        "--scenes",
        type    = int,
        default = 4,
        choices = [1, 2, 3, 4],
        help    = "Number of scenes to generate (default: 4)",
    )
    parser.add_argument(
        "--cpu",
        action  = "store_true",
        help    = "Force CPU inference (slow but works without GPU)",
    )
    parser.add_argument(
        "--output-dir",
        type    = str,
        default = None,
        help    = "Override output directory (default: Desktop)",
    )
    parser.add_argument(
        "--keep-temp",
        action  = "store_true",
        help    = "Keep temporary working files after completion",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print_banner()
    args = parse_args()

    # ── Pre-flight checks ────────────────────────────────────────────────────
    print("── Pre-flight checks ──────────────────────────────────────────")
    check_ffmpeg_binary()

    device = torch.device("cpu") if args.cpu else detect_device()

    # ── Output path ──────────────────────────────────────────────────────────
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = get_desktop_path()

    output_path = out_dir / OUTPUT_FILENAME
    print(f"  ✔ Output path  : {output_path}\n")

    # ── Temporary working directory ──────────────────────────────────────────
    work_dir = Path(tempfile.mkdtemp(prefix="locallens_ep1_"))
    print(f"  ✔ Work dir     : {work_dir}\n")

    try:
        # ── Build scene prompts ──────────────────────────────────────────────
        scene_prompts = EPISODE_SCENE_PROMPTS[: args.scenes]
        if args.prompt:
            # Append user prompt as additional style guidance
            scene_prompts = [f"{p}, {args.prompt}" for p in scene_prompts]
            print(f"  ✔ Custom prompt appended: {args.prompt}\n")

        print("── Scene Prompts ───────────────────────────────────────────────")
        for i, sp in enumerate(scene_prompts, 1):
            print(f"  Scene {i}: {sp[:90]}…")
        print()

        # ── Step 1: Load SD pipeline ─────────────────────────────────────────
        pipe = load_sd_pipeline(device)

        # ── Step 2: Generate images ──────────────────────────────────────────
        image_paths = generate_images(pipe, scene_prompts, work_dir, device)

        # Free VRAM before OpenCV work
        del pipe
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # ── Step 3: Animate clips ────────────────────────────────────────────
        clip_paths = create_animated_clips(image_paths, work_dir)

        # ── Step 4: Voiceover ────────────────────────────────────────────────
        audio_path = generate_voiceover(VOICEOVER_SCRIPT, work_dir)

        # ── Step 5: Assemble ─────────────────────────────────────────────────
        assemble_video(clip_paths, audio_path, output_path, work_dir)

        # ── Done ─────────────────────────────────────────────────────────────
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  ✅  VIDEO GENERATION COMPLETE                               ║")
        print(f"║  📁  {str(output_path):<56}║")
        print("╚══════════════════════════════════════════════════════════════╝\n")

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Generation cancelled by user.")
        sys.exit(1)

    except Exception as exc:
        print(f"\n[FATAL ERROR] {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        if not args.keep_temp:
            shutil.rmtree(work_dir, ignore_errors=True)
            print(f"  🗑  Temp files cleaned: {work_dir}")
        else:
            print(f"  📂 Temp files kept   : {work_dir}")


if __name__ == "__main__":
    main()
