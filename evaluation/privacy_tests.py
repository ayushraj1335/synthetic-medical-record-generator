"""Empirical leakage probes. These do not prove differential privacy."""
from evaluation.fidelity import duplicate_metrics, nearest_neighbor_distances

def run_empirical_privacy_checks(real, synthetic):
    return {'duplicate_check':duplicate_metrics(real,synthetic),
            'nearest_neighbor_check':nearest_neighbor_distances(real,synthetic),
            'limitation':'Empirical privacy tests can find some memorization but cannot establish or prove differential privacy.'}
