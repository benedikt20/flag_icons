import flagpy as fp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pandas as pd
import numpy as np
from PIL import Image

# Load ship and normalize to [0,1]
ship = mpimg.imread("data/ship.png")
if ship.dtype == np.uint8:
    ship = ship.astype(float) / 255.0
ship = np.rot90(ship, k=-1)
ship_h, ship_w = ship.shape[:2]

# Mask for black areas
mask = np.mean(ship[:, :, :3], axis=2) < 0.1

names = pd.read_csv("data/alpha2.csv")
df = fp.get_flag_df()#.head(5)

os.makedirs('flag_icons', exist_ok=True)

for country_name, row in df.iterrows():
    flag_img = row['flag']

    # Remove alpha channel if present
    if flag_img.shape[-1] == 4:
        flag_img = flag_img[:, :, :3]

    # Normalize flag to [0,1]
    if flag_img.dtype == np.uint8:
        flag_img = flag_img.astype(float) / 255.0

    # Resize flag to match ship shape using high-quality LANCZOS resampling
    pil_flag = Image.fromarray((flag_img * 255).astype(np.uint8))
    pil_flag_resized = pil_flag.resize((ship_w, ship_h), resample=Image.LANCZOS)
    flag_resized = np.array(pil_flag_resized).astype(float) / 255.0

    country_id = names.loc[names["country"] == country_name, "alpha2"].values[0]

    # Blend flag into black areas of ship
    result = ship.copy()
    for i in range(3):  # RGB
        result[:, :, i] = np.where(mask, flag_resized[:, :, i], result[:, :, i])

    # rotate back 90 degrees counter-clockwise
    result = np.rot90(result, k=1)

    out_path = os.path.join('flag_icons', f"{country_id}.png")
    #plt.imsave(out_path, result)
    result_uint8 = (result * 255).astype(np.uint8)
    Image.fromarray(result_uint8).save(out_path)
