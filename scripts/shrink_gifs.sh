#!/usr/bin/env bash
# Shrink the GIFs written by render_episodes.py for the README (keeps every 2nd frame, 224 px wide).
# Usage: bash scripts/shrink_gifs.sh docs/behaviors
set -euo pipefail
DIR="${1:-docs/behaviors}"
for f in "$DIR"/*.gif; do
  ffmpeg -loglevel error -y -i "$f" -vf "select='not(mod(n\,2))',setpts=N/(6.25*TB),fps=6.25,scale=224:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" "${f%.gif}.small.gif"
  mv "${f%.gif}.small.gif" "$f"
done
