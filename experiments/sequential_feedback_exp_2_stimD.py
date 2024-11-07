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
Extended experimental setup:

stim1, stim2 -> A <-> B <- stim3
                ^    
                |    
                v    
                C    

Experiment phases:
1. Initial assembly formation: Form stable assemblies for stim1 and stim2 in A
2. Association phase: Train stim3 in B to associate with stim2's assembly in A
3. Testing phase: Show stim1, then trigger with stim3 to cause assembly switching
"""

# Hyperparameters for assembly switching experiment
T = 0.95  # convergence threshold
ASSOCIATION_TRAINING_STEPS = 20  # steps to train stim3-stim2 association
SWITCH_TEST_REPEATS = 30  # number of times to test switching
PRE_SWITCH_STEPS = 10  # steps to show stim1 before attempting switch
POST_SWITCH_STEPS = 10  # steps to measure after introducing stim3


def is_converged(set_1: List, set_2: List) -> bool:
    intersection = set(set_1).intersection(set(set_2))
    percent_intersect = len(intersection) / len(set_2)
    print(f"Percent intersect: {percent_intersect}")
    if percent_intersect > T:
        return True
    return False


def silence_brain(b):
    b.winners = []
    b._new_winners = []
    b.saved_winners = []
    b.num_first_winners = -1
    return b

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

def train_stim3_association(b, stim2_complete, stim3_to_B, min_steps=10):
    """
    Train the association between stim3 in B and stim2's assembly in A
    by presenting both stimuli simultaneously.
    """
    combined_stim_map = {
        "areas_by_stim": {
            "stim2": ["A"],
            "stim3": ["B"]
        },
        "dst_areas_by_src_area": {
            "A": ["A", "B", "C"],
            "B": ["A", "B"],
            "C": ["A", "C"]
        }
    }
    
    for _ in range(ASSOCIATION_TRAINING_STEPS):
        b.project(combined_stim_map["areas_by_stim"], 
                 combined_stim_map["dst_areas_by_src_area"])

def test_assembly_switching(b, stim1_complete, stim3_to_B):
    """
    Test how stim3 triggers switching from stim1's assembly to stim2's assembly.
    Returns the sequence of assemblies in A during the switch.
    """
    assemblies_during_switch = []
    
    # First establish stim1's assembly
    for _ in range(PRE_SWITCH_STEPS):
        b.project(stim1_complete["areas_by_stim"], 
                 stim1_complete["dst_areas_by_src_area"])
        assemblies_during_switch.append(b.area_by_name["A"].saved_winners[-1])
    
    # Introduce stim3 to B and observe switching
    switch_map = {
        "areas_by_stim": {"stim3": ["B"]},
        "dst_areas_by_src_area": {
            "A": ["A", "B", "C"],
            "B": ["A", "B"],
            "C": ["A", "C"]
        }
    }
    
    for _ in range(POST_SWITCH_STEPS):
        b.project(switch_map["areas_by_stim"], 
                 switch_map["dst_areas_by_src_area"])
        assemblies_during_switch.append(b.area_by_name["A"].saved_winners[-1])
    
    return assemblies_during_switch

def main(n=100000, k=317, p=0.01, beta=0.04):
    # Initialize brain and areas
    b = brain.Brain(p, save_winners=True)
    b.add_stimulus("stim1", k)
    b.add_stimulus("stim2", k)
    b.add_stimulus("stim3", k)  # New stimulus that projects to B
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
    stim1_complete = {
        "areas_by_stim": {"stim1": ["A"]},
        "dst_areas_by_src_area": {
            "A": ["A", "B", "C"],
            "B": ["A", "B"],
            "C": ["A", "C"]
        }
    }

    stim2_complete = {
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
        stim1_complete["areas_by_stim"],
        stim1_complete["dst_areas_by_src_area"],
        target_area_name="A",
        min_steps=5
    )
    stim1_assemblies_A.append(b.area_by_name["A"].saved_winners[-1])
    stim1_assemblies_B.append(b.area_by_name["B"].saved_winners[-1])
    stim1_assemblies_C.append(b.area_by_name["C"].saved_winners[-1])

    project_until_converged(
        b,
        stim2_complete["areas_by_stim"],
        stim2_complete["dst_areas_by_src_area"],
        target_area_name="A",
        min_steps=5
    )
    stim2_assemblies_A.append(b.area_by_name["A"].saved_winners[-1])
    stim2_assemblies_B.append(b.area_by_name["B"].saved_winners[-1])
    stim2_assemblies_C.append(b.area_by_name["C"].saved_winners[-1])

    # Verify that stim1 and stim2 form distinct assemblies
    final_overlap = iou(stim1_assemblies_A[-1], stim2_assemblies_A[-1])
    print(f"Final overlap between stim1 and stim2 assemblies in A: {final_overlap}")
    if final_overlap > 0.5:
        print("Warning: stim1 and stim2 assemblies are too similar!")

    # New projection pattern for stim3 to B
    stim3_to_B = {
        "areas_by_stim": {"stim3": ["B"]},
        "dst_areas_by_src_area": {
            "B": ["A", "B"],
            "C": ["A", "C"],
            "A": ["A", "B", "C"]
        }
    }

    # Phase 1: Establish initial assemblies (your existing code)
    project_until_converged(
        b,
        stim3_to_B["areas_by_stim"],
        stim3_to_B["dst_areas_by_src_area"],
        target_area_name="B",
        min_steps=5
    )

    # Phase 2: Train stim3-stim2 association
    print("Training stim3-stim2 association...")
    train_stim3_association(b, stim2_complete, stim3_to_B)

    # Phase 3: Test assembly switching
    print("Testing assembly switching...")
    switching_results = []
    for test in tqdm(range(SWITCH_TEST_REPEATS)):
        b_copy = copy.deepcopy(b)
        assemblies_during_switch = test_assembly_switching(b_copy, stim1_complete, stim3_to_B)
        switching_results.append(assemblies_during_switch)

    # Analyze switching results
    stim2_reference = b.area_by_name["A"].saved_winners[-1]  # Reference stim2 assembly
    switching_similarity = []
    for trial in switching_results:
        similarity_curve = [iou(assembly, stim2_reference) for assembly in trial]
        switching_similarity.append(similarity_curve)

    # Plot switching results
    plt.figure(figsize=(10, 6))
    mean_similarity = np.mean(switching_similarity, axis=0)
    std_similarity = np.std(switching_similarity, axis=0)
    time_points = range(len(mean_similarity))
    
    plt.plot(time_points, mean_similarity, 'b-', label='Mean Similarity to Stim2 Assembly')
    plt.fill_between(time_points, 
                     mean_similarity - std_similarity,
                     mean_similarity + std_similarity,
                     alpha=0.3)
    
    plt.axvline(x=PRE_SWITCH_STEPS, color='r', linestyle='--', 
                label='Stim3 Introduction')
    plt.xlabel('Time Steps')
    plt.ylabel('Similarity to Stim2 Assembly (IoU)')
    plt.title('Assembly Switching from Stim1 to Stim2 via Stim3')
    plt.legend()
    plt.savefig("seq_feedback_2_assembly_switching_results.png")
    plt.close()

    # Save assemblies to pickle file
    assembly_data = {
        'stim1': {
            'A': stim1_assemblies_A,
            'B': stim1_assemblies_B,
            'C': stim1_assemblies_C
        },
        'stim2': {
            'A': stim2_assemblies_A,
            'B': stim2_assemblies_B, 
            'C': stim2_assemblies_C
        }
    }
    
    with open('sequential_feedback_assemblies.pkl', 'wb') as f:
        pickle.dump(assembly_data, f)


if __name__ == "__main__":
    main() 