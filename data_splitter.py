#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:39:19 2020

@author: d4ve
"""

import os, glob, re
from shutil import copyfile
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

p = Path("raw_data/*/*")
paths_images = glob.glob(str(p) + "*jpg")
cat = [int(re.search("\d+\.", i).group().strip(".")) for i in paths_images]

df = pd.DataFrame(list(zip(paths_images, cat)), columns=["path", "cat"])
df = df[df["cat"].map(df["cat"].value_counts()) > 350]


sets = ["train", "test"]

cats = [
    "kitchen",
    "facade",
    "bedroom",
    "bathroom",
    "garden",
    "pool",
    "terrace",
    "garage",
    "exterior",
    "details",
    "living_room",
    "hall",
    "reception",
    "balcony",
    "dining_room",
    "cafeteria",
]

df["cat"].replace([i for i in range(1, 17)], cats, inplace=True)

keep_cats = [
    'bedroom',
    'living_room',
    'kitchen',
    'bathroom']

#df = df[~df["cat"].str.contains("details")]

df = df[df["cat"].isin(keep_cats)]

# reduce size for benchmarking phase
#reduce, df = train_test_split(df, test_size=0.25)

train, test = train_test_split(df, test_size=0.20, random_state=26)

if not os.path.exists("data"):
    os.makedirs("data")

for s in sets:
    set_path = Path("data/" + s)
    if not os.path.exists(str(set_path)):
        os.makedirs(str(set_path))
    for cat in df['cat'].unique():
        cat_path = Path(str(set_path) + "/" + cat)
        if not os.path.exists(str(cat_path)):
            os.makedirs(str(cat_path))


def splitter(df, data_set):
    for i, image in enumerate(df.values):
        copyfile(image[0], Path(f"data/{data_set}/{image[1]}/{i}.jpg"))

splitter(train, "train")
splitter(test, "test")
