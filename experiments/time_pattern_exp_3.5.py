import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import brain
import brain_util as bu
import numpy as np
import random
import copy
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from typing import List


T = 0.95  # convergence threshold
FOLDS = 100  # num times to run train/test


def is_converged(set_1: List, set_2: List) -> bool:
    intersection = set(set_1).intersection(set(set_2))
    percent_intersect = len(intersection) / len(set_2)
    print(f"Percent intersect: {percent_intersect}")
    if percent_intersect > T:
        return True
    return False


def project_until_converged(b, stimuli, area_map, target_area_name: str, min_steps=5):
    i = 0
    while i < min_steps or not is_converged(
        b.area_by_name[target_area_name].saved_winners[-1],
        b.area_by_name[target_area_name].saved_winners[-2],
    ):
        b.project(stimuli, area_map)
        i += 1


def iou(set_1: List, set_2: List) -> float:
    intersection = set(set_1).intersection(set(set_2))
    union = set(set_1).union(set(set_2))
    return len(intersection) / len(union)

def silence_brain(b):
    b.winners = []
    b._new_winners = []
    b.saved_winners = []
    b.num_first_winners = -1
    return b



def main(n=100000, k=317, p=0.01, beta=0.01):

    b = brain.Brain(p, save_winners=True)

    b.add_stimulus("stim_1", k)
    b.add_stimulus("stim_2", k)
    b.add_stimulus("stim_3", k)
    b.add_stimulus("stim_4", k)

    b.add_area("A", n, k, beta)
    b.add_area("B", n, k, beta)
    b.add_area("C", n, k, beta)
    b.add_area("D", n, k, beta)

    b.project({"stim_1": ["A"]}, {})
    b.project({"stim_2": ["B"]}, {})
    b.project({"stim_3": ["C"]}, {})
    b.project({"stim_4": ["D"]}, {})


    stim_1_example = {
        "areas_by_stim": {"stim_1": ["A"]},
        "dst_areas_by_src_area": {"A":["A","B"]},
    }

    stim_2_example = {
        "areas_by_stim": {"stim_2": ["B"]},
        "dst_areas_by_src_area": {"B": ["B", "C"]},
    }

    stim_3_example = {
        "areas_by_stim": {"stim_3": ["C"]},
        "dst_areas_by_src_area": {"C": ["C", "D"]},
    }

    stim_4_example = {
        "areas_by_stim": {"stim_4": ["D"]},
        "dst_areas_by_src_area": {"D": ["D", "A"]},
    }

    associate_1_2_example = {
    "areas_by_stim": {"stim_1": ["A"], "stim_2": ["B"]},
    "dst_areas_by_src_area": {"A": ["A", "B"]},
    }

    associate_2_3_example = {
    "areas_by_stim": {"stim_2": ["B"], "stim_3": ["C"]},
    "dst_areas_by_src_area": {"B": ["B","C"]},
    }

    associate_3_4_example = {
    "areas_by_stim": {"stim_3": ["C"], "stim_4": ["D"]},
    "dst_areas_by_src_area": {"C": ["C","D"]},
    }

    associate_4_1_example = {
    "areas_by_stim": {"stim_4": ["D"], "stim_1": ["A"]},
    "dst_areas_by_src_area": {"D": ["D","A"]},
    }


    # Make assemblies:
    project_until_converged(
        b,
        stim_1_example["areas_by_stim"],
        stim_1_example["dst_areas_by_src_area"],
        target_area_name="A",
        min_steps=5,
    )
    stim_1_assembly = b.area_by_name["A"].saved_winners[-1]

    project_until_converged(
        b,
        stim_2_example["areas_by_stim"],
        stim_2_example["dst_areas_by_src_area"],
        target_area_name="B",
        min_steps=5,
    )
    stim_2_assembly = b.area_by_name["B"].saved_winners[-1]

    project_until_converged(
        b,
        stim_3_example["areas_by_stim"],
        stim_3_example["dst_areas_by_src_area"],
        target_area_name="C",
        min_steps=5,
    )
    stim_3_assembly = b.area_by_name["C"].saved_winners[-1]

    project_until_converged(
        b,
        stim_4_example["areas_by_stim"],
        stim_4_example["dst_areas_by_src_area"],
        target_area_name="D",
        min_steps=5,
    )
    stim_4_assembly = b.area_by_name["D"].saved_winners[-1]


    order = (["train_1_2"] * FOLDS + ["train_2_3"] * FOLDS + ["train_3_4"] * FOLDS + ["train_4_1"] * FOLDS)

    for index, o in tqdm(enumerate(order)):
        if o == "train_1_2":
            b.project(
                associate_1_2_example["areas_by_stim"],
                associate_1_2_example["dst_areas_by_src_area"],
            )
        elif o == "train_2_3":
            b.project(
                associate_2_3_example["areas_by_stim"],
                associate_2_3_example["dst_areas_by_src_area"],
            )
        elif o == "train_3_4":
            b.project(
                associate_3_4_example["areas_by_stim"],
                associate_3_4_example["dst_areas_by_src_area"],
            )
        elif o == "train_4_1":
            b.project(
                associate_4_1_example["areas_by_stim"],
                associate_4_1_example["dst_areas_by_src_area"],
            )
    
        
    A_winners_in_assembly_1 = []
    B_winners_in_assembly_2 = []
    C_winners_in_assembly_3 = []
    D_winners_in_assembly_4 = []
    firing_sequence = []

    for i in range(5):
        b.project({"stim_1": ["A"]}, {"A": ["A","B"], "B": ["B","C"], "C": ["C","D"], "D": ["D","A"]})
        A_winners_in_assembly_1.append(len(set(stim_1_assembly).intersection(set(b.area_by_name["A"].saved_winners[-1]))))
        B_winners_in_assembly_2.append(len(set(stim_2_assembly).intersection(set(b.area_by_name["B"].saved_winners[-1]))))
        C_winners_in_assembly_3.append(len(set(stim_3_assembly).intersection(set(b.area_by_name["C"].saved_winners[-1]))))
        D_winners_in_assembly_4.append(len(set(stim_4_assembly).intersection(set(b.area_by_name["D"].saved_winners[-1]))))

        if A_winners_in_assembly_1[-1] > B_winners_in_assembly_2[-1] and A_winners_in_assembly_1[-1] > C_winners_in_assembly_3[-1] and A_winners_in_assembly_1[-1] > D_winners_in_assembly_4[-1]:
            firing_sequence.append(1)
        elif B_winners_in_assembly_2[-1] > A_winners_in_assembly_1[-1] and B_winners_in_assembly_2[-1] > C_winners_in_assembly_3[-1] and B_winners_in_assembly_2[-1] > D_winners_in_assembly_4[-1]:
            firing_sequence.append(2)
        elif C_winners_in_assembly_3[-1] > A_winners_in_assembly_1[-1] and C_winners_in_assembly_3[-1] > B_winners_in_assembly_2[-1] and C_winners_in_assembly_3[-1] > D_winners_in_assembly_4[-1]:
            firing_sequence.append(3)
        elif D_winners_in_assembly_4[-1] > A_winners_in_assembly_1[-1] and D_winners_in_assembly_4[-1] > B_winners_in_assembly_2[-1] and D_winners_in_assembly_4[-1] > C_winners_in_assembly_3[-1]:
            firing_sequence.append(4)

    for i in range(15):
        b.project({}, {"A": ["A","B"], "B": ["B","C"], "C": ["C","D"], "D": ["D","A"]})
        A_winners_in_assembly_1.append(len(set(stim_1_assembly).intersection(set(b.area_by_name["A"].saved_winners[-1]))))
        B_winners_in_assembly_2.append(len(set(stim_2_assembly).intersection(set(b.area_by_name["B"].saved_winners[-1]))))
        C_winners_in_assembly_3.append(len(set(stim_3_assembly).intersection(set(b.area_by_name["C"].saved_winners[-1]))))
        D_winners_in_assembly_4.append(len(set(stim_4_assembly).intersection(set(b.area_by_name["D"].saved_winners[-1]))))

        if A_winners_in_assembly_1[-1] > B_winners_in_assembly_2[-1] and A_winners_in_assembly_1[-1] > C_winners_in_assembly_3[-1] and A_winners_in_assembly_1[-1] > D_winners_in_assembly_4[-1]:
            firing_sequence.append(1)
        elif B_winners_in_assembly_2[-1] > A_winners_in_assembly_1[-1] and B_winners_in_assembly_2[-1] > C_winners_in_assembly_3[-1] and B_winners_in_assembly_2[-1] > D_winners_in_assembly_4[-1]:
            firing_sequence.append(2)
        elif C_winners_in_assembly_3[-1] > A_winners_in_assembly_1[-1] and C_winners_in_assembly_3[-1] > B_winners_in_assembly_2[-1] and C_winners_in_assembly_3[-1] > D_winners_in_assembly_4[-1]:
            firing_sequence.append(3)
        elif D_winners_in_assembly_4[-1] > A_winners_in_assembly_1[-1] and D_winners_in_assembly_4[-1] > B_winners_in_assembly_2[-1] and D_winners_in_assembly_4[-1] > C_winners_in_assembly_3[-1]:
            firing_sequence.append(4)

    print(firing_sequence)

    time_steps = range(1, 21)
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, A_winners_in_assembly_1, label="A winners in assembly 1")
    plt.plot(time_steps, B_winners_in_assembly_2, label="B winners in assembly 2")
    plt.plot(time_steps, C_winners_in_assembly_3, label="C winners in assembly 3")
    plt.plot(time_steps, D_winners_in_assembly_4, label="D winners in assembly 4")

    plt.xlabel("time_steps")
    plt.ylabel("Winners")
    plt.title("Winners in areas A, B, C, and D that are in the formed assembly")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()




# Try just AB, BC, CD, DA, and then also try showing the stimulus like 5 times in a row.
# 