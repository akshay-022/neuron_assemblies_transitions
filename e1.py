import brain
import brain_util as bu
import numpy as np
import random
import copy
import pickle
import matplotlib.pyplot as plt

from collections import OrderedDict
from typing import List

T = 0.95


def is_converged(set_1: List, set_2: List) -> bool:
    intersection = set(set_1).intersection(set(set_2))
    percent_intersect = len(intersection) / len(set_2)
    print(f"Percent intersect: {percent_intersect}")
    if percent_intersect > T:
        return True
    return False


def e1(n=100000, k=317, p=0.01, beta=0.04):
    b = brain.Brain(p, save_winners=True)
    b.add_stimulus("stim", k)
    b.add_area("A", n, k, beta)
    b.add_area("B", n, k, beta)
    b.add_area("C", n, k, beta)
    b.add_area("D", n, k, beta)

    b.project({"stim": ["A"]}, {})
    b.project({"stim": ["B"]}, {})
    b.project({"stim": ["C"]}, {})

    # for i in range(100):
    saved_set = set()
    i = 0
    def project_until_converged(stimuli, area_map, max_steps=5):
        i = 0
        while i < max_steps or not is_converged(
            b.area_by_name["D"].saved_winners[-1], b.area_by_name["D"].saved_winners[-2]
        ):
            b.project(stimuli, area_map)
            i += 1

    project_until_converged({"stim": ["A"]}, {"A": ["A", "D"]})
    project_until_converged({"stim": ["B"]}, {"B": ["B", "D"]})
    project_until_converged({"stim": ["C"]}, {"C": ["C", "D"]})
    project_until_converged({"stim": ["A", "B", "C"]}, {"A": ["A", "D"], "B": ["B", "D"]})

    for i in range(3):
        b.project({"stim": ["A"]}, {"A": ["A"]})
        print(b.area_by_name["A"].w)
    b.area_by_name["A"].fix_assembly()
    for i in range(5):
        b.project({"stim": ["A"]}, {"A": ["A"]})
        print(b.area_by_name["A"].w)
    b.area_by_name["A"].unfix_assembly()
    for i in range(50):
        b.project({"stim": ["A"]}, {"A": ["A"]})
        print(b.area_by_name["A"].w)


# def explicit_assembly_test():
#     b = brain.Brain(0.5)
#     b.add_stimulus("stim", 3)
#     b.add_explicit_area("A", 10, 3, beta=0.5)
#     b.add_area("B", 10, 3, beta=0.5)

#     print(b.connectomes_by_stimulus["stim"]["A"])
#     print(b.connectomes["A"]["A"])
#     print(b.connectomes["A"]["B"].shape)
#     print(b.connectomes["B"]["A"].shape)

#     # Now test projection stimulus -> explicit area
#     print("Project stim->A")
#     b.project({"stim": ["A"]}, {})
#     print(b.area_by_name["A"].winners)
#     print(b.connectomes_by_stimulus["stim"]["A"])
#     # Now test projection stimulus, area -> area
#     b.project({"stim": ["A"]}, {"A": ["A"]})
#     print(b.area_by_name["A"].winners)
#     print(b.connectomes_by_stimulus["stim"]["A"])
#     print(b.connectomes["A"]["A"])

#     # project explicit A -> B
#     print("Project explicit A -> normal B")
#     b.project({}, {"A": ["B"]})
#     print(b.area_by_name["B"].winners)
#     print(b.connectomes["A"]["B"])
#     print(b.connectomes["B"]["A"])
#     print(b.connectomes_by_stimulus["stim"]["B"])


# def explicit_assembly_test2(rounds=20):
#     b = brain.Brain(0.1)
#     b.add_explicit_area("A", 100, 10, beta=0.5)
#     b.add_area("B", 10000, 100, beta=0.5)

#     b.area_by_name["A"].winners = list(range(10, 20))
#     b.area_by_name["A"].fix_assembly()
#     b.project({}, {"A": ["B"]})

#     # Test that if we fire back from B->A now, we don't recover the fixed assembly
#     b.area_by_name["A"].unfix_assembly()
#     b.project({}, {"B": ["A"]})
#     print(b.area_by_name["A"].winners)

#     b.area_by_name["A"].winners = list(range(10, 20))
#     b.area_by_name["A"].fix_assembly()
#     b.project({}, {"A": ["B"]})
#     for _ in range(rounds):
#         b.project({}, {"A": ["B"], "B": ["A", "B"]})
#         print(b.area_by_name["B"].w)

#     b.area_by_name["A"].unfix_assembly()
#     b.project({}, {"B": ["A"]})
#     print("After 1 B->A, got A winners:")
#     print(b.area_by_name["A"].winners)

#     for _ in range(4):
#         b.project({}, {"B": ["A"], "A": ["A"]})
#     print("After 5 B->A, got A winners:")
#     print(b.area_by_name["A"].winners)


def explicit_assembly_recurrent():
    b = brain.Brain(0.1)
    b.add_explicit_area("A", 100, 10, beta=0.5)

    b.area_by_name["A"].winners = list(range(60, 70))


if __name__ == "__main__":
    e1()
    # fixed_assembly_test()
    # explicit_assembly_test()
    # explicit_assembly_test2()
