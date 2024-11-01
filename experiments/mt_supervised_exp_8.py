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

"""
Experimental setup:

stim -> A <- reward

1. Learn stim -> A
2. Learn reward -> A
3. Learn (stim, reward) -> A
4. Compare iou(proj(stim), proj(reward)) vs iou(proj(control), proj(reward, A))

Result: 
iou(proj(stim), proj(reward)) > iou(proj(control), proj(reward, A))
for until convergence
lim_inf(iou(proj(stim), proj(reward))) == lim_inf(iou(proj(control), proj(reward, A)))

Todo:
- Figure out how get winners while assemblies are frozen
- Test the hypothesis that learning an association control -> reward_2 will keep it from learning control -> reward_1
- Time series input
"""

T = 0.95  # convergence threshold
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

def silence_brain(b):
    b.winners = []
    b._new_winners = []
    b.saved_winners = []
    b.num_first_winners = -1
    return b




def main(n=100000, k=317, p=0.01, beta=0.01):
    b = brain.Brain(p, save_winners=True)
    b.add_stimulus("stim", k)
    b.add_stimulus("reward1", k)
    b.add_stimulus("control", k)
    b.add_stimulus("reward2", k)
    b.add_area("A", n, k, beta)
    b.project({"stim": ["A"]}, {})
    b.project({"reward1": ["A"]}, {})
    b.project({"control": ["A"]}, {})
    b.project({"reward2": ["A"]}, {})
    stim_A_neurons = b.area_by_name["A"].saved_winners[-1]

    reward1_example = {
        "areas_by_stim": {"reward1": ["A"]},
        "dst_areas_by_src_area": {"A": ["A"]},
    }
    reward2_example = {
        "areas_by_stim": {"reward2": ["A"]},
        "dst_areas_by_src_area": {"A": ["A"]},
    }
    associate_example1 = {
        "areas_by_stim": {"stim": ["A"], "reward1": ["A"]},
        "dst_areas_by_src_area": {"A": ["A"]},
    }
    associate_example2 = {
        "areas_by_stim": {"control": ["A"], "reward2": ["A"]},
        "dst_areas_by_src_area": {"A": ["A"]},
    }
    stim_example = {
        "areas_by_stim": {"stim": ["A"]},
        "dst_areas_by_src_area": {"A": ["A"]},
    }
    control_example = {
        "areas_by_stim": {"control": ["A"]},
        "dst_areas_by_src_area": {"A": ["A"]},
    }

    # Learn the reward until convergence
    project_until_converged(
        b,
        reward1_example["areas_by_stim"],
        reward1_example["dst_areas_by_src_area"],
        target_area_name="A",
        min_steps=5,
    )
    reward_A = b.area_by_name["A"].saved_winners[-1]

    project_until_converged(
        b,
        reward2_example["areas_by_stim"],
        reward2_example["dst_areas_by_src_area"],
        target_area_name="A",
        min_steps=5,
    )
    # Learn the stimulus until convergence
    project_until_converged(
        b,
        stim_example["areas_by_stim"],
        stim_example["dst_areas_by_src_area"],
        target_area_name="A",
        min_steps=5,
    )

    project_until_converged(
        b,
        control_example["areas_by_stim"],
        control_example["dst_areas_by_src_area"],
        target_area_name="A",
        min_steps=5,
    )

    stim_A = b.area_by_name["A"].saved_winners[-1]

    associations = []
    stim_to_reward1 = []
    control_to_reward2 = []
    stim_to_reward2 = []
    control_to_reward1 = []

    order = (["train1"] * FOLDS )#+ ["train2"] * FOLDS
    
    stim_A_assemblies = []
    control_A_assemblies = []
    reward1_A_assemblies = []
    reward2_A_assemblies = []


    for index, o in tqdm(enumerate(order)):
        if o == "train1":
            b.project(
                associate_example1["areas_by_stim"],
                associate_example1["dst_areas_by_src_area"],
            )
            associations.append(b.area_by_name["A"].saved_winners[-1])
        
        

        # Every nth iteration, see what the assemblies are from scratch
        if index % 5 == 0:
            # b.area_by_name["A"].fix_assembly()
            b_copy = copy.deepcopy(b)
            b_copy = silence_brain(b_copy)
            project_until_converged(
                b_copy,
                stim_example["areas_by_stim"],
                stim_example["dst_areas_by_src_area"],
                target_area_name="A",
                min_steps=5,
            )
            stim_A_assemblies.append(b_copy.area_by_name["A"].saved_winners[-1])
            
            b_copy = copy.deepcopy(b)
            b_copy = silence_brain(b_copy)
            project_until_converged(
                b_copy,
                control_example["areas_by_stim"],
                control_example["dst_areas_by_src_area"],
                target_area_name="A",
                min_steps=5,
            )
            control_A_assemblies.append(b_copy.area_by_name["A"].saved_winners[-1])

            b_copy = copy.deepcopy(b)
            b_copy = silence_brain(b_copy)
            project_until_converged(
                b_copy,
                reward1_example["areas_by_stim"],
                reward1_example["dst_areas_by_src_area"],
                target_area_name="A",
                min_steps=5,
            )
            reward1_A_assemblies.append(b_copy.area_by_name["A"].saved_winners[-1])

            b_copy = copy.deepcopy(b)
            b_copy = silence_brain(b_copy)
            project_until_converged(
                b_copy,
                reward2_example["areas_by_stim"],
                reward2_example["dst_areas_by_src_area"],
                target_area_name="A",
                min_steps=5,
            )
            reward2_A_assemblies.append(b_copy.area_by_name["A"].saved_winners[-1])

        
    stim_reward1_overlap = [iou(stim_A_assemblies[i], reward1_A_assemblies[i]) for i in range(len(stim_A_assemblies))]
    control_reward2_overlap = [iou(control_A_assemblies[i], reward2_A_assemblies[i]) for i in range(len(control_A_assemblies))]
    output_index = list(range(len(stim_reward1_overlap)))[::FOLDS]

    colors = {"Control to Reward 1": "r", "Stimulus to reward 1": "b", "Stimulus to reward 2": "g", "Control to reward 2": "k"}
    # plot consistency
    plt.plot(stim_reward1_overlap, label="Stimulus to Reward 1", color='r')
    plt.plot(control_reward2_overlap, label="Control to Reward 2", color='b')
    plt.plot(
        output_index,
        control_reward2_overlap,
        label="Control to Reward 2",
        color='g',
    )
    plt.plot(
        output_index,
        control_reward2_overlap,
        label="Stimulus to Reward 2",
        color='k',
    )

    legend_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", label=k, markerfacecolor=v, markersize=10
        )
        for k, v in colors.items()
    ]
    plt.legend(handles=legend_handles)
    plt.savefig("iou_9.png")
    print


if __name__ == "__main__":
    main()
