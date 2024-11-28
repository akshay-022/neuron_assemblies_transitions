import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import brain
import brain_util as bu
import numpy as np
import random
import copy
import pickle
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from typing import List
from matplotlib import pyplot as plt
import seaborn as sns

"""

"""

T = 0.95  # convergence threshold
FOLDS = 100  # num times to run train/test
K = 6


def is_converged(set_1: List, set_2: List) -> bool:
    intersection = set(set_1).intersection(set(set_2))
    percent_intersect = len(intersection) / len(set_2)
    print(f"Percent intersect: {percent_intersect}")
    if percent_intersect > T:
        return True
    return False


def iou(set_1: List, set_2: List) -> float:
    intersection = set(set_1).intersection(set(set_2))
    union = set(set_1).union(set(set_2))
    return len(intersection) / len(union)


def silence_brain(b, areas: list):
    for a in areas:
        # b.area_by_name[a].saved_w = []
        # b.area_by_name[a].winners = []
        b.area_by_name[a]._new_winners = []
        b.area_by_name[a].saved_winners = []
    return b


def brain_init(b, areas, n, k, beta):
    for area in areas:
        b.add_area(area, n, k, beta)
    b.project({"i1": ["A"]}, {})
    b.project({"i2": ["B"]}, {})
    b.project({"r1": ["C"]}, {})
    b.project({"r2": ["C"]}, {})
    return b


def project_until_converged(
    b,
    areas_by_stim=None,
    dst_areas_by_src_area=None,
    target_area_name=None,
    min_steps=5,
):
    i = 0
    while i < min_steps or not is_converged(
        b.area_by_name[target_area_name].saved_winners[-1],
        b.area_by_name[target_area_name].saved_winners[-2],
    ):
        b.project(areas_by_stim, dst_areas_by_src_area)
        i += 1
        # print(i)


def main(n=100000, k=317, p=0.01, beta=0.02):
    areas = ["A", "B", "C"]

    b = brain.Brain(p, save_winners=True)
    b.add_stimulus("i1", k)
    b.add_stimulus("i2", k)
    b.add_stimulus("r1", k)
    b.add_stimulus("r2", k)
    b = brain_init(b, areas, n, k, beta)

    i1_example = {
        "areas_by_stim": {"i1": ["A"]},
        "dst_areas_by_src_area": {"A": ["A", "C"], "C": ["C"]},
    }
    i2_example = {
        "areas_by_stim": {"i2": ["B"]},
        "dst_areas_by_src_area": {"B": ["B", "C"], "C": ["C"]},
    }
    i1_r1_example = {
        "areas_by_stim": {"i1": ["A"], "r1": ["C"]},
        "dst_areas_by_src_area": {"A": ["A", "C"], "C": ["C"]},
    }
    i2_r2_example = {
        "areas_by_stim": {"i2": ["B"], "r2": ["C"]},
        "dst_areas_by_src_area": {"B": ["B", "C"], "C": ["C"]},
    }
    r1_example = {
        "areas_by_stim": {"r1": ["C"]},
        "dst_areas_by_src_area": {"C": ["C"]},
    }
    r2_example = {
        "areas_by_stim": {"r2": ["C"]},
        "dst_areas_by_src_area": {"C": ["C"]},
    }

    # # Learn the reward until convergence
    project_until_converged(b, **i1_r1_example, target_area_name="C")
    # r1_ref_assembly = b.area_by_name["C"].saved_winners[-1]
    b = silence_brain(b, areas)
    project_until_converged(b, **i2_r2_example, target_area_name="C")
    # r2_ref_assembly = b.area_by_name["C"].saved_winners[-1]
    b = silence_brain(b, areas)

    # set beta to 0 in all areas for testing
    for area in areas:
        b.area_by_name[area].beta = 0.0

    project_until_converged(b, **r1_example, target_area_name="C")
    r1_ref_assembly = b.area_by_name["C"].saved_winners[-1]
    b = silence_brain(b, areas)

    project_until_converged(b, **r2_example, target_area_name="C")
    r2_ref_assembly = b.area_by_name["C"].saved_winners[-1]
    b = silence_brain(b, areas)

    project_until_converged(b, **i1_example, target_area_name="C", min_steps=2)
    i1_ref_assembly = b.area_by_name["C"].saved_winners[-1]
    b = silence_brain(b, areas)

    project_until_converged(b, **i2_example, target_area_name="C", min_steps=2)
    i2_ref_assembly = b.area_by_name["C"].saved_winners[-1]
    b = silence_brain(b, areas)

    # check iou of each assembly
    i1_r1_iou = iou(i1_ref_assembly, r1_ref_assembly)
    i1_r2_iou = iou(i1_ref_assembly, r2_ref_assembly)
    i2_r1_iou = iou(i2_ref_assembly, r1_ref_assembly)
    i2_r2_iou = iou(i2_ref_assembly, r2_ref_assembly)

    print(f"iou(i1, r1): {i1_r1_iou}")
    print(f"iou(i1, r2): {i1_r2_iou}")
    print(f"iou(i2, r1): {i2_r1_iou}")
    print(f"iou(i2, r2): {i2_r2_iou}")

    # print a confusion matrix
    # Create confusion matrix data
    data = np.array([[i1_r1_iou, i1_r2_iou],
                     [i2_r1_iou, i2_r2_iou]])

    # Create labels
    stimuli = ['i1', 'i2']
    responses = ['r1', 'r2']

    # Create pandas DataFrame
    df_cm = pd.DataFrame(data, index=stimuli, columns=responses)

    # Plot the heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(df_cm, annot=True, fmt='.2f', cmap='Blues')

    # Add title and labels
    plt.title('IOU Confusion Matrix')
    plt.ylabel('Stimuli')
    plt.xlabel('Responses')

    # Save the figure
    plt.savefig('iou_confusion_matrix.png')
    plt.close()

    print


if __name__ == "__main__":
    main()
