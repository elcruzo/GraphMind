#!/usr/bin/env python3
"""
Statistical analysis for GraphMind experiments
"""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy import stats
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Run statistical tests on results')
    parser.add_argument('--data', type=str, default='results/', help='Data directory')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')
    args = parser.parse_args()
    
    print(f"Running statistical tests on data from {args.data}")
    print(f"Significance level: {args.alpha}")
    
    # TODO: Implement actual statistical tests based on result format
    # Placeholder for now
    
    # Example: Dummy t-test
    group1 = np.random.normal(0.85, 0.05, 30)
    group2 = np.random.normal(0.82, 0.05, 30)
    
    t_stat, p_value = stats.ttest_ind(group1, group2)
    
    print(f"\nT-test Results:")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significant: {'Yes' if p_value < args.alpha else 'No'}")
    
    # Save results
    results = {
        'test': 't-test',
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < args.alpha,
        'alpha': args.alpha
    }
    
    output_path = Path(args.data) / 'statistical_tests.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")

if __name__ == '__main__':
    main()