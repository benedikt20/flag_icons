#------------------------------------------------------------------------------
# Written by: Benedikt Fadel, benediktfadel@gmail.com
# Date: 2025-07-30
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# This script is used to make flag icons for plotting.
# - Uses an base icon to extract the mask of the flags
# - Generates an icon from each flag and blends it into the base icon
# - Saves the icon flags in an output directory
#
# Flags downloaded from:
#   https://stefangabos.github.io/world_countries/
#------------------------------------------------------------------------------
# Usage:
# python make_flag_icons.py
#------------------------------------------------------------------------------
#
#
#
#
#
# Imports
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import os
from PIL import Image
import shutil


# NOTE: SE is renamed to SW to match the AIS flag code, add other name changes here
name_changes = {
    "SE": "SW" # Sweden name change
}

# Directories
FLAG_DIR = "data/128x96" # Standard flag directory
ICON_PATH = "data/ship.png" # Icon image to use for the flags
OUTPUT_DIR = "flag_icons" # Output directory

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load ship and normalize to [0,1]
ship = mpimg.imread(ICON_PATH)
if ship.dtype == np.uint8:
    ship = ship.astype(float) / 255.0
ship = np.rot90(ship, k=-1)
ship_h, ship_w = ship.shape[:2]

# Mask for black areas
mask = np.mean(ship[:, :, :3], axis=2) < 0.1

# Get list of PNG files (Alpha-2 codes)
flag_files = [f.replace('.png', '').lower() for f in os.listdir(FLAG_DIR) if f.endswith('.png')]
print(f"Found {len(flag_files)} flag PNGs in {FLAG_DIR}")

# Process each flag
for flag_file in flag_files:
    alpha_2 = flag_file.upper()  # Convert to uppercase for consistency
    flag_path = os.path.join(FLAG_DIR, f"{flag_file}.png")

    if alpha_2 in name_changes:
        # Rename to SW
        print(f"Renaming {alpha_2} to {name_changes[alpha_2]}")
        alpha_2 = name_changes[alpha_2]
    
    try:
        # Load flag image
        flag_img = mpimg.imread(flag_path)
        
        # Remove alpha channel if present
        if flag_img.shape[-1] == 4:
            flag_img = flag_img[:, :, :3]
        
        # Normalize flag to [0,1]
        if flag_img.dtype == np.uint8:
            flag_img = flag_img.astype(float) / 255.0
        
        # Resize flag to match ship shape using LANCZOS resampling
        pil_flag = Image.fromarray((flag_img * 255).astype(np.uint8))
        pil_flag_resized = pil_flag.resize((ship_w, ship_h), resample=Image.LANCZOS)
        flag_resized = np.array(pil_flag_resized).astype(float) / 255.0
        
        # Blend flag into black areas of ship
        result = ship.copy()
        for i in range(3):  # RGB
            result[:, :, i] = np.where(mask, flag_resized[:, :, i], result[:, :, i])
        
        # Rotate back 90 degrees counter-clockwise
        result = np.rot90(result, k=1)
        
        # Save output
        out_path = os.path.join(OUTPUT_DIR, f"{alpha_2}.png")
        result_uint8 = (result * 255).astype(np.uint8)
        Image.fromarray(result_uint8).save(out_path)
        #print(f"Saved blended flag for {alpha_2} to {out_path}")
        
    except Exception as e:
        print(f"Error processing {alpha_2}: {e}")

# print the number of flags in the output directory
print(f"Number of flags in {OUTPUT_DIR}: {len(os.listdir(OUTPUT_DIR))}")



