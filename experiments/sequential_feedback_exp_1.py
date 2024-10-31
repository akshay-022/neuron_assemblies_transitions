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

stim1, stim2 -> A <-> B
                ^    
                |    
                v    
                C    

1. Project stim1 -> A -> (B,C) -> A until convergence
2. Project stim2 -> A -> (B,C) -> A until convergence
3. Compare assemblies in A, B, C for both stimuli

Todo:
- Analyze stability of assemblies
- Test different projection patterns

All inter area connections are always on. Show stim1 only first, form assembly. Show stim2 only then, form assembly. Then keep showing them one after the other in time and see how their assemblies differ/change.
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


def main(n=100000, k=317, p=0.01, beta=0.04):
    # Initialize brain and areas
    b = brain.Brain(p, save_winners=True)
    b.add_stimulus("stim1", k)
    b.add_stimulus("stim2", k)
    b.add_area("A", n, k, beta)
    b.add_area("B", n, k, beta)
    b.add_area("C", n, k, beta)

    # Initial projections to establish baseline assemblies
    b.project({"stim1": ["A"]}, {})
    b.project({"stim2": ["A"]}, {})

    # Define projection patterns
    # First only project from A to B and C (no feedback yet)
    stim1_forward = {
        "areas_by_stim": {"stim1": ["A"]},
        "dst_areas_by_src_area": {
            "A": ["A", "B", "C"],
        }
    }

    stim2_forward = {
        "areas_by_stim": {"stim2": ["A"]},
        "dst_areas_by_src_area": {
            "A": ["A", "B", "C"],
        }
    }

    # Full projection patterns with feedback
    stim1_example = {
        "areas_by_stim": {"stim1": ["A"]},
        "dst_areas_by_src_area": {
            "A": ["A", "B", "C"],
            "B": ["A", "B"],
            "C": ["A", "C"]
        }
    }

    stim2_example = {
        "areas_by_stim": {"stim2": ["A"]},
        "dst_areas_by_src_area": {
            "A": ["A", "B", "C"],
            "B": ["A", "B"],
            "C": ["A", "C"]
        }
    }

    # Track assemblies for each stimulus
    stim1_assemblies_A = []
    stim1_assemblies_B = []
    stim1_assemblies_C = []
    stim2_assemblies_A = []
    stim2_assemblies_B = []
    stim2_assemblies_C = []

    # First establish assemblies in all areas using forward projections, no feedback otherwise brain throws error
    project_until_converged(
        b,
        stim1_forward["areas_by_stim"],
        stim1_forward["dst_areas_by_src_area"],
        target_area_name="A",
        min_steps=5
    )
    stim1_assemblies_A.append(b.area_by_name["A"].saved_winners[-1])
    stim1_assemblies_B.append(b.area_by_name["B"].saved_winners[-1])
    stim1_assemblies_C.append(b.area_by_name["C"].saved_winners[-1])

    project_until_converged(
        b,
        stim2_forward["areas_by_stim"],
        stim2_forward["dst_areas_by_src_area"],
        target_area_name="A",
        min_steps=5
    )
    stim2_assemblies_A.append(b.area_by_name["A"].saved_winners[-1])
    stim2_assemblies_B.append(b.area_by_name["B"].saved_winners[-1])
    stim2_assemblies_C.append(b.area_by_name["C"].saved_winners[-1])



    # Then establish assemblies in all areas using forward and feedback projections
    project_until_converged(
        b,
        stim1_example["areas_by_stim"],
        stim1_example["dst_areas_by_src_area"],
        target_area_name="A",
        min_steps=5
    )
    stim1_assemblies_A.append(b.area_by_name["A"].saved_winners[-1])
    stim1_assemblies_B.append(b.area_by_name["B"].saved_winners[-1])
    stim1_assemblies_C.append(b.area_by_name["C"].saved_winners[-1])

    project_until_converged(
        b,
        stim2_example["areas_by_stim"],
        stim2_example["dst_areas_by_src_area"],
        target_area_name="A",
        min_steps=5
    )
    stim2_assemblies_A.append(b.area_by_name["A"].saved_winners[-1])
    stim2_assemblies_B.append(b.area_by_name["B"].saved_winners[-1])
    stim2_assemblies_C.append(b.area_by_name["C"].saved_winners[-1])

    # Now run alternating projections with feedback
    for fold in tqdm(range(FOLDS)):
        # First show stimulus 1 for 5 iterations
        for _ in range(5):
            b.project(stim1_example["areas_by_stim"], stim1_example["dst_areas_by_src_area"])
        for _ in range(5):
            b.project(stim2_example["areas_by_stim"], stim2_example["dst_areas_by_src_area"])
            
        stim1_assemblies_A.append(b.area_by_name["A"].saved_winners[-1])
        stim1_assemblies_B.append(b.area_by_name["B"].saved_winners[-1])
        stim1_assemblies_C.append(b.area_by_name["C"].saved_winners[-1])
        stim2_assemblies_A.append(b.area_by_name["A"].saved_winners[-1])
        stim2_assemblies_B.append(b.area_by_name["B"].saved_winners[-1])
        stim2_assemblies_C.append(b.area_by_name["C"].saved_winners[-1])

    # Calculate overlaps between assemblies
    stim1_stim2_B_overlap = [iou(a, b) for a, b in zip(stim1_assemblies_B, stim2_assemblies_B)]
    stim1_stim2_C_overlap = [iou(a, c) for a, c in zip(stim1_assemblies_C, stim2_assemblies_C)]

    # Plot results
    colors = {"stim1_AB": "b", "stim1_AC": "g", "stim2_AB": "r", "stim2_AC": "purple"}
    
    plt.figure(figsize=(10, 6))
    plt.plot(stim1_stim2_B_overlap, label="Stim1 Stim2 B Overlap", color=colors["stim1_AB"])
    plt.plot(stim1_stim2_C_overlap, label="Stim1 Stim2 C Overlap", color=colors["stim1_AC"])

    plt.xlabel("Iterations")
    plt.ylabel("Intersection over Union")
    plt.title("Assembly Overlap Evolution")
    plt.legend()
    plt.savefig("iou_seq_feedback_1.png")
    plt.close()


if __name__ == "__main__":
    main() 