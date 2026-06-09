import os
import pickle
import numpy as np
import pandas as pd

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(MODULE_DIR, "data")

# 1. The hidden helper 
def _fetch_file(relative_path):
    full_path = os.path.join(DATA_DIR, relative_path)
    if full_path.endswith('.pkl'):
        with open(full_path, "rb") as f:
            return pickle.load(f)
    elif full_path.endswith('.npy'):
        return np.load(full_path)
    elif full_path.endswith('.csv'):
        return pd.read_csv(full_path, index_col=0)

# 2. Your dictionary-based user functions
def load_simulation():
    return {
        "contour": _fetch_file("Simulation_file/contour_simulation_list_24groups_2880.pkl"),
        "images": _fetch_file("Simulation_file/image_simulation_list_24groups_2880.npy"),
        "labels": _fetch_file("Simulation_file/label_simulation_list_24groups_2880.npy")
    }

def load_leaf():
    return {
        "contour": _fetch_file("contour_leaf_v2.pkl"),
        "images": _fetch_file("image_leaf_v2.npy"),
        "labels": _fetch_file("label_leaf_v2.npy")
    }

def load_mpeg7():
    return {
        "contour": _fetch_file("contour_mpeg7_v2.pkl"),
        "images": _fetch_file("image_mpeg7_v2.npy"),
        "labels": _fetch_file("label_mpeg7_v2.npy")
    }

def load_healthy_bmmc():
    return {
        "contour": _fetch_file("Healthy_BMMC_monocytes_contours.pkl"),
        "metadata": _fetch_file("Healthy_BMMC_monocytes_metadata.csv")
    }