import numpy as np
from time import time


def compute_statistics(best_performances, avg_performances, glob_best_change):
    # P_best
    best_performances = np.array(best_performances)
    best_partial = np.average(best_performances, axis=0)  # axis=0 means vertically.
    p_best = np.sum(best_partial)

    # R_best
    # best_partial[i] contains P_best[i].
    best_robusteness = np.zeros(999)

    for g in range(999):
        x = best_partial[g] / best_partial[g + 1]
        if x > 1:
            best_robusteness[g] = 1
        else:
            best_robusteness[g] = x

    r_best = np.sum(best_robusteness)

    # P_avg
    avg_performances = np.array(avg_performances)
    avg_partial = np.average(avg_performances, axis=0)  # axis=0 means vertically.
    p_avg = np.sum(avg_partial)

    # R_avg
    # avg_partial[i] contains P_avg[i].
    avg_robusteness = np.zeros(999)

    for g in range(999):
        x = avg_partial[g] / avg_partial[g + 1]
        if x > 1:
            avg_robusteness[g] = 1
        else:
            avg_robusteness[g] = x
    r_avg = np.sum(avg_robusteness)

    # change in global min
    glob_best_change = np.array(glob_best_change)
    gbc_partial = np.average(glob_best_change)

    best_performances = best_partial
    avg_performances = avg_partial
    best_robusteness = best_robusteness
    avg_robusteness = avg_robusteness
    return best_performances, avg_performances, best_robusteness, avg_robusteness


