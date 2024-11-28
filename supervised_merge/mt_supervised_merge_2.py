import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import brain
import numpy as np

import pandas as pd
from tqdm import tqdm
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
        b.area_by_name[a]._new_winners = []
        b.area_by_name[a].saved_winners = []
    return b


def brain_init(b, areas, n, k, beta, num_classes):
    for area in areas:
        b.add_area(area, n, k, beta)
    # Project stimuli to their respective areas
    for i in range(1, num_classes + 1):
        area = f'Area{i}'
        b.project({f"i{i}": [area]}, {})
    # Project rewards to area Z
    for i in range(1, num_classes + 1):
        b.project({f"r{i}": ["Z"]}, {})
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


def main(n=100000, k=317, p=0.01, beta=0.02, num_classes=10):
    areas = [f'Area{i}' for i in range(1, num_classes + 1)] + ['Z']

    b = brain.Brain(p, save_winners=True)

    # Add stimuli 'i1' to 'iN' and 'r1' to 'rN'
    for i in range(1, num_classes + 1):
        b.add_stimulus(f"i{i}", k)
        b.add_stimulus(f"r{i}", k)

    b = brain_init(b, areas, n, k, beta, num_classes)

    # Define examples
    examples = {}
    for i in range(1, num_classes + 1):
        area = f'Area{i}'
        examples[f'i{i}_r{i}_example'] = {
            'areas_by_stim': {f'i{i}': [area], f'r{i}': ['Z']},
            'dst_areas_by_src_area': {area: [area, 'Z'], 'Z': ['Z']},
        }
        examples[f'i{i}_example'] = {
            'areas_by_stim': {f'i{i}': [area]},
            'dst_areas_by_src_area': {area: [area, 'Z'], 'Z': ['Z']},
        }
        examples[f'r{i}_example'] = {
            'areas_by_stim': {f'r{i}': ['Z']},
            'dst_areas_by_src_area': {'Z': ['Z']},
        }

    # Learn the reward until convergence
    r_ref_assemblies = {}
    i_ref_assemblies = {}
    for i in range(1, num_classes + 1):
        b = silence_brain(b, areas)
        project_until_converged(
            b, **examples[f'i{i}_r{i}_example'], target_area_name='Z'
        )
        # r_ref_assemblies[f'r{i}'] = b.area_by_name['Z'].saved_winners[-1]
        b = silence_brain(b, areas)

    # Set beta to 0 in all areas for testing
    for area in areas:
        b.area_by_name[area].beta = 0.0

    # Get reference assemblies for 'r{i}'
    for i in range(1, num_classes + 1):
        b = silence_brain(b, areas)
        project_until_converged(
            b, **examples[f'r{i}_example'], target_area_name='Z'
        )
        r_ref_assemblies[f'r{i}'] = b.area_by_name['Z'].saved_winners[-1]

    # Get reference assemblies for 'i{i}'
    for i in range(1, num_classes + 1):
        b = silence_brain(b, areas)
        project_until_converged(
            b,
            **examples[f'i{i}_example'],
            target_area_name='Z',
            min_steps=2,
        )
        i_ref_assemblies[f'i{i}'] = b.area_by_name['Z'].saved_winners[-1]

    # Compute IOUs between each i_ref and r_ref
    data = np.zeros((num_classes, num_classes))
    for i in range(1, num_classes + 1):
        for j in range(1, num_classes + 1):
            iou_value = iou(
                i_ref_assemblies[f'i{i}'], r_ref_assemblies[f'r{j}']
            )
            data[i - 1, j - 1] = iou_value
            print(f"iou(i{i}, r{j}): {iou_value:.3f}")

    # Create labels
    stimuli = [f'i{i}' for i in range(1, num_classes + 1)]
    responses = [f'r{j}' for j in range(1, num_classes + 1)]

    # Create pandas DataFrame
    df_cm = pd.DataFrame(data, index=stimuli, columns=responses)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt='.3f', cmap='Blues')

    # Add title and labels
    plt.title('IOU Confusion Matrix')
    plt.ylabel('Stimuli')
    plt.xlabel('Responses')

    # Save the figure
    plt.savefig(f'iou_confusion_matrix_size_{num_classes}.png')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--num_classes',
        type=int,
        default=10,
        help='number of stimulus/reward/areas',
    )
    args = parser.parse_args()
    num_classes = args.num_classes
    main(num_classes=num_classes)
