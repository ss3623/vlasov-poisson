from firedrake import *
import numpy as np

def compute_relative_l2_error(moment1, moment2):
    """Compute relative L2 error between two moments"""
    return norm(moment1 - moment2) / norm(moment2)

def analyze_convergence(M_values, moment_ref, moment_list):
    """Compute errors for all M values"""
    errors = []
    for i in range(len(M_values)):
        error = compute_relative_l2_error(moment_list[i], moment_ref)
        errors.append(error)
    return errors