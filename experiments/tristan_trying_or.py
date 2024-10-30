import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

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

"""
Experimental setup:

stim_1 -> A <- reward
stim_2 -> A <- reward


1. Learn stim_1 -> A
2. Learn stim_2 -> A
3. Learn reward -> A
4. Learn reward -> A
5. Learn (stim_1, reward) -> A
6. Learn (stim_2, reward) -> A
7. Compare iou(proj(stim_1), proj(reward)) and iou(proj(stim_2), proj(reward)) vs iou(proj(control), proj(reward, A))
8. Try to implement something that checks if the stimulus was stim_1 OR stim_2 by defining a threshold
   and seeing if the activation in the reward assembly was above or below it when exposed to different stimuli.

"""

T = 0.95  # convergence threshold
Z = 3  # num train examples per fold
FOLDS = 150  # num times to run train/test


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


def main(n=100000, k=317, p=0.01, beta=0.01):
    b = brain.Brain(p, save_winners=True)
    b.add_stimulus("stim_1", k)
    b.add_stimulus("stim_2", k)
    b.add_stimulus("reward", k)
    b.add_stimulus("control", k)
    b.add_area("A", n, k, beta)
    b.project({"stim_1": ["A"]}, {})
    b.project({"stim_2": ["A"]}, {})
    b.project({"reward": ["A"]}, {})
    b.project({"control": ["A"]}, {})
    stim_A_neurons = b.area_by_name["A"].saved_winners[-1]

    stim_1_example = {
        "areas_by_stim": {"stim_1": ["A"]},
        "dst_areas_by_src_area": {"A": ["A"]},
    }
    stim_2_example = {
        "areas_by_stim": {"stim_2": ["A"]},
        "dst_areas_by_src_area": {"A": ["A"]},
    }
    reward_example = {
        "areas_by_stim": {"reward": ["A"]},
        "dst_areas_by_src_area": {"A": ["A"]},
    }
    control_example = {
        "areas_by_stim": {"control": ["A"]},
        "dst_areas_by_src_area": {"A": ["A"]},
    }
    associate_1_example = {
        "areas_by_stim": {"stim_1": ["A"], "reward": ["A"]},
        "dst_areas_by_src_area": {"A": ["A"]},
    }
    associate_2_example = {
        "areas_by_stim": {"stim_2": ["A"], "reward": ["A"]},
        "dst_areas_by_src_area": {"A": ["A"]},
    }

    # Learn the reward until convergence
    project_until_converged(
        b,
        reward_example["areas_by_stim"],
        reward_example["dst_areas_by_src_area"],
        target_area_name="A",
        min_steps=5,
    )
    reward_A = b.area_by_name["A"].saved_winners[-1]

    # Learn stimulus 1 until convergence
    project_until_converged(
        b,
        stim_1_example["areas_by_stim"],
        stim_1_example["dst_areas_by_src_area"],
        target_area_name="A",
        min_steps=5,
    )
    stim1_A = b.area_by_name["A"].saved_winners[-1]

    # Learn stimulus 2 until convergence
    project_until_converged(
        b,
        stim_2_example["areas_by_stim"],
        stim_2_example["dst_areas_by_src_area"],
        target_area_name="A",
        min_steps=5,
    )
    stim2_A = b.area_by_name["A"].saved_winners[-1]

    associations_1 = []
    associations_2 = []
    stim1_to_reward = []
    stim2_to_reward = []
    control_to_reward = []

    order = (["train_1"] * Z + ["train_2"] * Z + ["stim_1", "stim_2", "control"]) * FOLDS
    for o in tqdm(order):
        if o == "train_1":
            b.project(
                associate_1_example["areas_by_stim"],
                associate_1_example["dst_areas_by_src_area"],
            )
            associations_1.append(b.area_by_name["A"].saved_winners[-1])

        if o == "train_2":
            b.project(
                associate_2_example["areas_by_stim"],
                associate_2_example["dst_areas_by_src_area"],
            )
            associations_2.append(b.area_by_name["A"].saved_winners[-1])

        # b.area_by_name["A"].fix_assembly()
        if o == "stim_1":
            b.project(
                stim_1_example["areas_by_stim"],
                stim_1_example["dst_areas_by_src_area"],
            )
            stim1_to_reward.append(b.area_by_name["A"].saved_winners[-1])

        if o == "stim_2":
            b.project(
                stim_2_example["areas_by_stim"],
                stim_2_example["dst_areas_by_src_area"],
            )
            stim2_to_reward.append(b.area_by_name["A"].saved_winners[-1])

        if o == "control":
            b.project(
                control_example["areas_by_stim"],
                control_example["dst_areas_by_src_area"],
            )
            control_to_reward.append(b.area_by_name["A"].saved_winners[-1])
            # b.area_by_name["A"].unfix_assembly()

    stim1_overlap = [iou(stim1_A, a) for a in associations_1]
    stim2_overlap = [iou(stim2_A, a) for a in associations_2]
    reward1_overlap = [iou(reward_A, a) for a in associations_1]
    reward2_overlap = [iou(reward_A, a) for a in associations_2]

    stim1_to_reward_overlap = [iou(reward_A, a) for a in stim1_to_reward]
    stim2_to_reward_overlap = [iou(reward_A, a) for a in stim2_to_reward]
    control_to_reward_overlap = [iou(reward_A, a) for a in control_to_reward]

    output_index = list(range(len(stim1_overlap)))[::Z]

    # colors = {"reward1": "b", "reward2": "g", "stim1": "r", "stim2": "c", "exp1": "m", "exp2": "y", "control": "k"}
    colors = {"exp1": "r", "exp2": "b", "control": "k"}
    # plot consistency

    # I don't think we care as much about these:
    # plt.plot(stim1_overlap, label="Stimulus_1", color=colors["stim1"])
    # plt.plot(stim2_overlap, label="Stimulus_2", color=colors["stim2"])
    # plt.plot(reward1_overlap, label="Reward 1", color=colors["reward1"])
    # plt.plot(reward2_overlap, label="Reward 2", color=colors["reward2"])
    
    plt.plot(
        output_index,
        control_to_reward_overlap,
        label="Control to Reward",
        color=colors["control"],
    )
    plt.plot(
        output_index,
        stim1_to_reward_overlap,
        label="Stimulus 1 to Reward",
        color=colors["exp1"],
    )

    plt.plot(
        output_index,
        stim2_to_reward_overlap,
        label="Stimulus 2 to Reward",
        color=colors["exp2"],
    )

    legend_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", label=k, markerfacecolor=v, markersize=10
        )
        for k, v in colors.items()
    ]
    plt.legend(handles=legend_handles)
    plt.savefig("iou_7.png")
    print


if __name__ == "__main__":
    main()
