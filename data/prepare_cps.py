#!/usr/bin/env python3
"""
Prepare CPS ASEC data for the dp-statsmodels wage regression example.

This script downloads and processes CPS ASEC 2024 data from the Census Bureau.

Usage:
    python data/prepare_cps.py

Output:
    data/cps_asec_2024_wages.csv
"""

import os
import sys
import zipfile
import urllib.request
import tempfile
import pandas as pd
import numpy as np

# CPS ASEC 2024 (March supplement, reference year 2023)
CPS_URL = "https://www2.census.gov/programs-surveys/cps/datasets/2024/march/asecpub24csv.zip"
PERSON_FILE = "pppub24.csv"
OUTPUT_FILE = "data/cps_asec_2024_wages.csv"


def download_cps():
    """Download CPS ASEC ZIP file."""
    print(f"Downloading CPS ASEC from {CPS_URL}...")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        urllib.request.urlretrieve(CPS_URL, tmp.name)
        return tmp.name


def extract_person_file(zip_path):
    """Extract person-level CSV from ZIP."""
    print(f"Extracting {PERSON_FILE}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with tempfile.TemporaryDirectory() as tmpdir:
            zf.extract(PERSON_FILE, tmpdir)
            return os.path.join(tmpdir, PERSON_FILE)


def hga_to_years(hga):
    """Convert CPS A_HGA education recode to years of schooling."""
    if pd.isna(hga):
        return np.nan
    hga = int(hga)
    if hga <= 31:  # None or preschool
        return 0
    elif hga <= 38:  # Grades 1-12 no diploma
        return max(0, hga - 31)
    elif hga == 39:  # HS grad or GED
        return 12
    elif hga == 40:  # Some college, no degree
        return 14
    elif hga == 41:  # Associate degree (academic)
        return 14
    elif hga == 42:  # Associate degree (vocational)
        return 14
    elif hga == 43:  # Bachelor's degree
        return 16
    elif hga == 44:  # Master's degree
        return 18
    elif hga == 45:  # Professional degree
        return 20
    elif hga == 46:  # Doctorate
        return 22
    else:
        return 12  # Default to HS


def process_cps(csv_path):
    """Process CPS data for wage regression."""
    print("Processing CPS data...")

    # Load only needed columns
    cols = ['A_AGE', 'A_SEX', 'WSAL_VAL', 'A_HGA', 'WKSWORK', 'A_FNLWGT', 'PEARNVAL']
    df = pd.read_csv(csv_path, usecols=cols)
    print(f"  Total records: {len(df):,}")

    # Filter to working-age adults (25-64) with positive wage/salary income
    df_workers = df[
        (df['A_AGE'] >= 25) &
        (df['A_AGE'] <= 64) &
        (df['WSAL_VAL'] > 0) &
        (df['WKSWORK'] >= 10)  # At least 10 weeks worked
    ].copy()
    print(f"  Workers (25-64, positive wages, 10+ weeks): {len(df_workers):,}")

    # Create education years
    df_workers['educ_years'] = df_workers['A_HGA'].apply(hga_to_years)

    # Create potential experience
    df_workers['experience'] = df_workers['A_AGE'] - df_workers['educ_years'] - 6
    df_workers['experience'] = df_workers['experience'].clip(lower=0)
    df_workers['experience_sq'] = df_workers['experience'] ** 2

    # Create female indicator (A_SEX: 1=male, 2=female)
    df_workers['female'] = (df_workers['A_SEX'] == 2).astype(int)

    # Log wages
    df_workers['log_wage'] = np.log(df_workers['WSAL_VAL'])

    # Select and clean final columns
    final_cols = ['log_wage', 'educ_years', 'experience', 'experience_sq',
                  'female', 'WSAL_VAL', 'A_FNLWGT', 'A_AGE']
    df_final = df_workers[final_cols].dropna()
    print(f"  Final sample: {len(df_final):,}")

    return df_final


def main():
    # Ensure we're in repo root
    if not os.path.exists('data'):
        os.makedirs('data')

    # Download
    zip_path = download_cps()

    try:
        # Extract and process
        with zipfile.ZipFile(zip_path, 'r') as zf:
            with tempfile.TemporaryDirectory() as tmpdir:
                zf.extract(PERSON_FILE, tmpdir)
                csv_path = os.path.join(tmpdir, PERSON_FILE)
                df = process_cps(csv_path)
    finally:
        os.unlink(zip_path)

    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")

    # Summary
    print(f"\n=== Data Summary ===")
    print(f"Sample size: {len(df):,}")
    print(f"Wage skewness: {df['WSAL_VAL'].skew():.2f}")
    print(f"Mean log wage: {df['log_wage'].mean():.3f}")
    print(f"Mean education: {df['educ_years'].mean():.1f} years")
    print(f"Female share: {df['female'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
