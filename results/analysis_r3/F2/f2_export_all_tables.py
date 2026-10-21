#!/usr/bin/env python3
"""f2_export_all_tables.py

Generates all markdown tables and CSV artifacts for UNIT=F2.
"""

import os
import numpy as np
import pandas as pd
from f2_matched_comparison import run_matched_comparison
from f2_protocol_fair_comparison import compute_protocol_comparisons
from f2_tempsr_investigation import investigate_tempsr


def export():
    # 1. Matched comparison
    summary_df, per_fold_dict = run_matched_comparison()
    summary_df.to_csv("f2_matched_summary.csv", index=False)

    # Save per-fold tables
    for setting in ["primary", "secondary"]:
        for ds, df in per_fold_dict[setting].items():
            df.to_csv(f"f2_folds_{setting}_{ds}.csv", index=False)

    # 2. Protocol comparison
    proto_df = compute_protocol_comparisons()
    proto_df.to_csv("f2_protocol_comparison.csv", index=False)

    # 3. TempSR analysis
    tempsr_df = investigate_tempsr()
    tempsr_df.to_csv("f2_tempsr_analysis.csv", index=False)

    print("All CSV tables successfully exported to out/F2.")


if __name__ == "__main__":
    export()
