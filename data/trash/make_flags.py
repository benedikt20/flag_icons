import flagpy as fp
import matplotlib.pyplot as plt
import os
import pandas as pd

names = pd.read_csv("data/alpha2.csv")

df = fp.get_flag_df()

# Create the 'flags' directory if it doesn't exist
os.makedirs('flags', exist_ok=True)

for country_name, row in df.iterrows():
    flag_img = row['flag']
    country_id = names.loc[names["country"] == country_name, "alpha2"].values[0]

    filename = os.path.join('flags', f"{country_id}.png")
    plt.imsave(filename, flag_img)