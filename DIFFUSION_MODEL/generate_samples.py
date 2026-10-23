"""
Utility script to generate sample digits (0-9) in static/images/samples/
for instant 1-click testing in the web app UI.
"""

import os
from PIL import Image, ImageDraw
import numpy as np

os.makedirs("static/images/samples", exist_ok=True)

# Procedurally generate clean, centered 28x28 grayscale digits
for digit in range(10):
    img = Image.new("L", (28, 28), color=0)
    draw = ImageDraw.Draw(img)
    cx, cy = 14, 14

    if digit == 0:
        draw.ellipse([cx - 7, cy - 10, cx + 7, cy + 10], outline=255, width=2)
    elif digit == 1:
        draw.line([cx, cy - 10, cx, cy + 10], fill=255, width=3)
        draw.line([cx - 3, cy - 7, cx, cy - 10], fill=255, width=2)
        draw.line([cx - 5, cy + 10, cx + 5, cy + 10], fill=255, width=2)
    elif digit == 2:
        draw.arc([cx - 7, cy - 10, cx + 7, cy], start=180, end=360, fill=255, width=2)
        draw.line([cx + 7, cy, cx - 7, cy + 10], fill=255, width=2)
        draw.line([cx - 7, cy + 10, cx + 7, cy + 10], fill=255, width=3)
    elif digit == 3:
        draw.arc([cx - 6, cy - 10, cx + 6, cy], start=270, end=90, fill=255, width=2)
        draw.arc([cx - 6, cy, cx + 6, cy + 10], start=270, end=90, fill=255, width=2)
        draw.line([cx, cy, cx + 3, cy], fill=255, width=2)
    elif digit == 4:
        draw.line([cx - 6, cy - 10, cx - 6, cy + 1], fill=255, width=2)
        draw.line([cx - 6, cy + 1, cx + 7, cy + 1], fill=255, width=2)
        draw.line([cx + 3, cy - 10, cx + 3, cy + 10], fill=255, width=2)
    elif digit == 5:
        draw.line([cx + 6, cy - 10, cx - 6, cy - 10], fill=255, width=2)
        draw.line([cx - 6, cy - 10, cx - 6, cy - 1], fill=255, width=2)
        draw.arc([cx - 7, cy - 2, cx + 6, cy + 10], start=270, end=90, fill=255, width=2)
        draw.line([cx - 6, cy - 1, cx, cy - 1], fill=255, width=2)
    elif digit == 6:
        draw.arc([cx - 7, cy - 10, cx + 7, cy + 10], start=90, end=270, fill=255, width=2)
        draw.ellipse([cx - 7, cy - 1, cx + 7, cy + 10], outline=255, width=2)
    elif digit == 7:
        draw.line([cx - 7, cy - 10, cx + 7, cy - 10], fill=255, width=2)
        draw.line([cx + 7, cy - 10, cx - 4, cy + 10], fill=255, width=2)
    elif digit == 8:
        draw.ellipse([cx - 6, cy - 10, cx + 6, cy - 1], outline=255, width=2)
        draw.ellipse([cx - 7, cy - 1, cx + 7, cy + 10], outline=255, width=2)
    elif digit == 9:
        draw.ellipse([cx - 7, cy - 10, cx + 7, cy + 1], outline=255, width=2)
        draw.line([cx + 7, cy - 10, cx + 7, cy + 10], fill=255, width=2)

    # Save 28x28 sample
    out_path = f"static/images/samples/digit_{digit}.png"
    img.save(out_path)
    print(f"Saved {out_path}")

print("All sample digits generated!")
