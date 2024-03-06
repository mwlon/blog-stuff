ffmpeg -i results/interp/earth_%5d.png \
  -r 12 \
  -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=white" \
  -pix_fmt yuv420p \
  -profile:v main \
  results/spectrum.mp4
