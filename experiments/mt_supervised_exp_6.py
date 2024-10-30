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
Randomize training order
Figure out how get winners while assemblies are frozen

"""
T = 0.95
Z = 3
FOLDS = 100


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


def e1(n=100000, k=317, p=0.01, beta=0.01):
    b = brain.Brain(p, save_winners=True)
    b.add_stimulus("stim", k)
    b.add_stimulus("reward", k)
    b.add_stimulus("control", k)
    # b.add_stimulus("RA", k)
    # b.add_stimulus("RB", k)
    b.add_area("A", n, k, beta)
    # b.add_area("B", n, k, beta)
    # b.add_area("D", n, k, beta)

    b.project({"stim": ["A"]}, {})
    b.project({"reward": ["A"]}, {})
    b.project({"control": ["A"]}, {})
    stim_A_neurons = b.area_by_name["A"].saved_winners[-1]

    reward_example = {
        "areas_by_stim": {"reward": ["A"]},
        "dst_areas_by_src_area": {"A": ["A"]},
    }
    associate_example = {
        "areas_by_stim": {"stim": ["A"], "reward": ["A"]},
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
        reward_example["areas_by_stim"],
        reward_example["dst_areas_by_src_area"],
        target_area_name="A",
        min_steps=5,
    )
    reward_A = b.area_by_name["A"].saved_winners[-1]

    # Learn the stimulus until convergence
    project_until_converged(
        b,
        stim_example["areas_by_stim"],
        stim_example["dst_areas_by_src_area"],
        target_area_name="A",
        min_steps=5,
    )
    stim_A = b.area_by_name["A"].saved_winners[-1]

    associations = []
    stim_to_reward = []
    control_to_reward = []
    for i in tqdm(range(FOLDS)):
        for _ in range(Z):
            b.project(
                associate_example["areas_by_stim"],
                associate_example["dst_areas_by_src_area"],
            )
            associations.append(b.area_by_name["A"].saved_winners[-1])

        # b.area_by_name["A"].fix_assembly()
        b.project(
            stim_example["areas_by_stim"],
            stim_example["dst_areas_by_src_area"],
        )
        stim_to_reward.append(b.area_by_name["A"].saved_winners[-1])

        b.project(
            control_example["areas_by_stim"],
            control_example["dst_areas_by_src_area"],
        )
        control_to_reward.append(b.area_by_name["A"].saved_winners[-1])
        # b.area_by_name["A"].unfix_assembly()

    stim_overlap = [iou(stim_A, a) for a in associations]
    reward_overlap = [iou(reward_A, a) for a in associations]
    stim_to_reward_overlap = [iou(reward_A, a) for a in stim_to_reward]
    control_to_reward_overlap = [iou(reward_A, a) for a in control_to_reward]
    output_index = list(range(len(stim_overlap)))[::Z]

    colors = {"reward": "r", "stim": "b", "exp": "g", "control": "k"}
    # plot consistency
    plt.plot(stim_overlap, label="Stimulus", color=colors["stim"])
    plt.plot(reward_overlap, label="Reward", color=colors["reward"])
    plt.plot(
        output_index,
        control_to_reward_overlap,
        label="Control to Reward",
        color=colors["control"],
    )
    plt.plot(
        output_index,
        stim_to_reward_overlap,
        label="Stimulus to Reward",
        color=colors["exp"],
    )

    legend_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", label=k, markerfacecolor=v, markersize=10
        )
        for k, v in colors.items()
    ]
    plt.legend(handles=legend_handles)
    plt.savefig("iou_6.png")
    print


if __name__ == "__main__":
    e1()
