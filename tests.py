import brain
import brain_util as bu
import numpy as np
import random
import copy
import pickle
import matplotlib.pyplot as plt

from collections import OrderedDict


def fixed_assembly_test(n=100000, k=317, p=0.01, beta=0.01):
    b = brain.Brain(p)
    b.add_stimulus("stim", k)
    b.add_area("A", n, k, beta)
    b.project({"stim": ["A"]}, {})
    for i in range(3):
        b.project({"stim": ["A"]}, {"A": ["A"]})
        print(b.area_by_name["A"].w)
    b.area_by_name["A"].fix_assembly()
    for i in range(5):
        b.project({"stim": ["A"]}, {"A": ["A"]})
        print(b.area_by_name["A"].w)
    b.area_by_name["A"].unfix_assembly()
    for i in range(5):
        b.project({"stim": ["A"]}, {"A": ["A"]})
        print(b.area_by_name["A"].w)


def explicit_assembly_test():
    b = brain.Brain(0.5)
    b.add_stimulus("stim", 3)
    b.add_explicit_area("A", 10, 3, beta=0.5)
    b.add_area("B", 10, 3, beta=0.5)

    print(b.connectomes_by_stimulus["stim"]["A"])
    print(b.connectomes["A"]["A"])
    print(b.connectomes["A"]["B"].shape)
    print(b.connectomes["B"]["A"].shape)

    # Now test projection stimulus -> explicit area
    print("Project stim->A")
    b.project({"stim": ["A"]}, {})
    print(b.area_by_name["A"].winners)
    print(b.connectomes_by_stimulus["stim"]["A"])
    # Now test projection stimulus, area -> area
    b.project({"stim": ["A"]}, {"A": ["A"]})
    print(b.area_by_name["A"].winners)
    print(b.connectomes_by_stimulus["stim"]["A"])
    print(b.connectomes["A"]["A"])

    # project explicit A -> B
    print("Project explicit A -> normal B")
    b.project({}, {"A": ["B"]})
    print(b.area_by_name["B"].winners)
    print(b.connectomes["A"]["B"])
    print(b.connectomes["B"]["A"])
    print(b.connectomes_by_stimulus["stim"]["B"])


def explicit_assembly_test2(rounds=20):
    b = brain.Brain(0.1)
    b.add_explicit_area("A", 100, 10, beta=0.5)
    b.add_area("B", 10000, 100, beta=0.5)

    b.area_by_name["A"].winners = list(range(10, 20))
    b.area_by_name["A"].fix_assembly()
    b.project({}, {"A": ["B"]})

    # Test that if we fire back from B->A now, we don't recover the fixed assembly
    b.area_by_name["A"].unfix_assembly()
    b.project({}, {"B": ["A"]})
    print(b.area_by_name["A"].winners)

    b.area_by_name["A"].winners = list(range(10, 20))
    b.area_by_name["A"].fix_assembly()
    b.project({}, {"A": ["B"]})
    for _ in range(rounds):
        b.project({}, {"A": ["B"], "B": ["A", "B"]})
        print(b.area_by_name["B"].w)

    b.area_by_name["A"].unfix_assembly()
    b.project({}, {"B": ["A"]})
    print("After 1 B->A, got A winners:")
    print(b.area_by_name["A"].winners)

    for _ in range(4):
        b.project({}, {"B": ["A"], "A": ["A"]})
    print("After 5 B->A, got A winners:")
    print(b.area_by_name["A"].winners)


def explicit_assembly_recurrent():
    b = brain.Brain(0.1)
    b.add_explicit_area("A", 100, 10, beta=0.5)

    b.area_by_name["A"].winners = list(range(60, 70))


if __name__ == "__main__":
    # fixed_assembly_test()
    # explicit_assembly_test()
    explicit_assembly_test2()
