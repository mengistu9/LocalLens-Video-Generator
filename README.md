# 🎬 LocalLens: Episodic Video Generator

AI-powered text-to-video generator that creates cinematic episodic content using Stable Diffusion, OpenCV, and FFmpeg.

## ✨ Features

- 🖼️ **AI Image Generation** - 4 scene images via Stable Diffusion
- 🎥 **Ken Burns Animation** - Zoom/pan effects with OpenCV
- 🎤 **AI Voiceover** - Text-to-speech narration
- 📹 **Video Assembly** - FFmpeg-powered MP4 output
- 📱 **Vertical Format** - 1080x1920 for YouTube Shorts
- 🎮 **GPU/CPU Support** - Auto-detection with fallback
- 🌍 **Cross-Platform** - Windows, Mac, Linux

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (system package)
# Windows: winget install ffmpeg
# Mac: brew install ffmpeg
# Linux: sudo apt install ffmpeg

# Generate video
python locallens.py
