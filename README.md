# BeatFunCreator

[![Ko-fi](https://img.shields.io/badge/Ko--fi-Support%20this%20project-ff5e5b?logo=ko-fi&logoColor=white)](https://ko-fi.com/phantomvertigo)

A desktop application for creating `.funscript` files from audio/video files. Automatically detects beats in music and generates synchronized haptic feedback scripts for devices like The Handy.

<img width="1535" height="1030" alt="preview" src="https://github.com/user-attachments/assets/888b2b29-096f-472b-b386-2f5e2a3d3257" />

## Features

- **Beat Detection** — Automatically detect beats from audio with adjustable threshold, bass-only filtering, and minimum gap control
- **Tempo Grid** — Detect or manually set BPM with time signature support, subdivisions, and visual beat grid overlay
- **FunScript Generation** — Generate movement scripts in Peak or Alternating mode with configurable speed, position range, and noise
- **Visual Editor** — Interactive waveform, movement graph, and override zones with full zoom/pan support
- **Keyframe Editing** — Fine-tune individual keyframe positions by dragging, add independent keyframes, or reset modifications
- **Override Zones** — Define regions with different generation settings (threshold, speed, mode, etc.)
- **Work Area Markers** — In/Out markers restrict all operations to a specific time range; areas outside are dimmed
- **Beat Intensity** — Set intensity per beat (0-100%) with 5 quick presets, affecting stroke amplitude
- **Video Preview** — Sync a video preview window to the playhead for visual reference
- **Playback** — Play audio with adjustable speed (0.1x-3.0x), playhead snapping to beats or peaks
- **Multi-Beat Placement** — Place multiple evenly-spaced beats at once with fixed or tempo-based spacing
- **Undo/Redo** — Full undo/redo support for all editing operations
- **Project Save/Load** — Save and restore complete project state including beats, zones, tempo, keyframes, and file paths
- **Drag & Drop** — Drop audio/video files directly onto the window (requires tkinterdnd2)

## Installation

### Requirements

- Python 3.8 or newer

### Quick Start

1. Download or clone this repository
2. Run `install.bat` to install dependencies
3. Run `start.bat` to launch the application

### Manual Installation

```bash
pip install -r requirements.txt
python beatfuncreator.py
```

### Dependencies

| Package | Purpose |
|---|---|
| librosa | Audio analysis and beat/tempo detection |
| numpy | Numerical computation |
| scipy | Signal processing (peak detection) |
| sounddevice | Audio playback |
| matplotlib | Waveform and movement visualization |
| opencv-python | Video frame preview *(optional)* |
| tkinterdnd2 | Drag-and-drop file support *(optional)* |

## Usage

1. **Load a file** — Click Browse or drag an audio/video file onto the window
2. **Set work area** — Adjust the In/Out markers to define the region to work on
3. **Analyze beats** — Adjust the threshold slider and click "Analyze Beats"
4. **Fine-tune** — Edit beats manually: click to add, right-click to delete, drag to move
5. **Generate funscript** — Configure movement settings and click "Create FunScript"
6. **Save project** — Save your work to resume later

### Keyboard Shortcuts

| Key | Action |
|---|---|
| Space | Play / Pause |
| Home / End | Jump to In / Out marker |
| Left / Right | Step playhead by 1 second |
| 1-5 | Recall beat intensity preset |
| Delete | Delete selected beats/keyframes |
| Ctrl+Z / Ctrl+Y | Undo / Redo |
| Ctrl+C / Ctrl+V | Copy / Paste beats or keyframes |
| Ctrl+A | Select all beats |

## Supported Formats

**Audio:** MP3, WAV, FLAC, OGG, M4A, AAC, WMA

**Video:** MP4, AVI, MKV, MOV, WMV, FLV, WEBM

## Troubleshooting

- **"Missing dependencies"** — Run `install.bat` first
- **"Python is not installed"** — Install Python from [python.org](https://www.python.org/downloads/) and make sure to check "Add to PATH"
- **Video button shows error** — Run `pip install opencv-python`
- **No drag-and-drop** — Run `pip install tkinterdnd2`
- **Reset settings** — Run `reset_config.bat` to restore default configuration

## Support

If you find this tool useful, consider supporting its development:

[![Ko-fi](https://img.shields.io/badge/Ko--fi-Support%20this%20project-ff5e5b?logo=ko-fi&logoColor=white)](https://ko-fi.com/phantomvertigo)

## License

Free for personal use. Modification and redistribution are not permitted without permission. See [LICENSE](LICENSE) for details.
