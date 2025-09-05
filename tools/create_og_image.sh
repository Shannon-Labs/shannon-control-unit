#!/bin/bash

# Create a professional OG image for Shannon Labs SCU
# Size: 1200x630 (standard OG image dimensions)

OUTPUT="web/assets/shannon-scu-og-image.png"

# Create the image with gradient background and text
convert -size 1200x630 \
  -define gradient:angle=135 \
  gradient:'#0052E0'-'#001833' \
  -gravity North \
  -pointsize 28 \
  -fill '#4A9FFF' \
  -font Helvetica-Bold \
  -annotate +0+60 'SHANNON LABS' \
  -gravity North \
  -pointsize 72 \
  -fill white \
  -font Helvetica-Bold \
  -annotate +0+120 'Shannon Control Unit' \
  -gravity North \
  -pointsize 36 \
  -fill '#B8D4FF' \
  -font Helvetica \
  -annotate +0+220 'Cruise Control for LLM Training' \
  -gravity Center \
  -pointsize 48 \
  -fill white \
  -font Helvetica-Bold \
  -annotate +0+20 '10.6% BPT Improvement' \
  -gravity Center \
  -pointsize 32 \
  -fill '#B8D4FF' \
  -annotate +0+80 'Validated at 3B Scale' \
  -gravity South \
  -pointsize 24 \
  -fill '#4A9FFF' \
  -annotate +0+60 'shannonlabs.dev' \
  -gravity SouthEast \
  -pointsize 20 \
  -fill '#4A9FFF' \
  -annotate +30+30 'Patent Pending' \
  "$OUTPUT"

echo "Created OG image at: $OUTPUT"
echo "Dimensions: 1200x630px"