#!/usr/bin/env python
"""
MULTI-VIEW MIMIC-CXR PREPROCESSING PIPELINE
============================================
Creates train/val/test CSV files for multi-view chest X-ray report generation.

Features:
- Groups images by study_id for multi-view samples
- Extracts and cleans FINDINGS + IMPRESSION from reports
- Properly aligns view_positions with image_paths (no NaN filtering bug)
- Verifies no data leakage between splits
- Saves both CSV and Parquet formats

Usage:
    python preprocess_multiview.py

Output:
    - multiview_train.csv / .parquet
    - multiview_val.csv / .parquet  
    - multiview_test.csv / .parquet
"""

import os
import pandas as pd
from tqdm import tqdm
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

# Data paths - UPDATE THESE TO YOUR PATHS
META_PATH = 'mimic-cxr-dataset/metadata.csv'
BASE_IMAGE_DIR = 'mimic-cxr-dataset/official_data_iccv_final/files'
BASE_REPORT_DIR = 'mimic-cxr-dataset/mimic-cxr-reports/files'

# Multi-view configuration
MULTIVIEW_CONFIG = {
    'view_preferences': ['PA', 'LATERAL', 'AP'],  # Priority order for view selection
    'min_views_per_study': 1,                      # Minimum views required
    'max_views_per_study': 3,                      # Maximum views to include
    'allow_single_view': True,                     # Include single-view studies
    'require_pa': False,                           # Require PA view in each study
}

# Split configuration
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Output configuration
OUTPUT_DIR = '.'  # Current directory
SAVE_PARQUET = True  # Also save as parquet (recommended for large datasets)


# ============================================
# PATH HELPER FUNCTIONS
# ============================================

def get_jpg_path(subject_id, study_id, dicom_id):
    """Construct path to JPEG image file."""
    subject_folder = f"p{str(subject_id)}"
    top_folder = f"p{str(subject_id)[:2]}"
    study_folder = f"s{str(study_id)}"
    return os.path.join(BASE_IMAGE_DIR, top_folder, subject_folder, study_folder, dicom_id + '.jpg')


def get_report_path(subject_id, study_id):
    """Construct path to radiology report file."""
    subject_folder = f"p{str(subject_id)}"
    top_folder = f"p{str(subject_id)[:2]}"
    report_filename = f"s{study_id}.txt"
    return os.path.join(BASE_REPORT_DIR, top_folder, subject_folder, report_filename)


# ============================================
# TEXT PREPROCESSING
# ============================================

def format_clinical_note(report_text):
    """
    Extract and clean FINDINGS and IMPRESSION sections from radiology report.
    
    Args:
        report_text: Raw report text
        
    Returns:
        Cleaned combined text (findings + impression)
    """
    # Extract FINDINGS section
    findings_match = re.search(
        r"FINDINGS:\s*([\s\S]*?)(?=\n(?:IMPRESSION|EXAMINATION|INDICATION|TECHNIQUE|COMPARISON):|$)",
        report_text,
        re.IGNORECASE
    )
    
    # Extract IMPRESSION section
    impression_match = re.search(
        r"IMPRESSION:\s*([\s\S]*?)(?=\n(?:EXAMINATION|INDICATION|TECHNIQUE|COMPARISON):|$)",
        report_text,
        re.IGNORECASE
    )

    findings_text = findings_match.group(1).strip() if findings_match else ""
    impression_text = impression_match.group(1).strip() if impression_match else ""

    # Combine and clean
    combined_text = f"{findings_text} {impression_text}".strip()
    combined_text = re.sub(r'\n+', ' ', combined_text)      # Remove newlines
    combined_text = re.sub(r'\s{2,}', ' ', combined_text)   # Remove extra spaces
    combined_text = re.sub(r'[^\w\s.,?!-]', '', combined_text)  # Keep only safe chars
    combined_text = combined_text.strip()

    return combined_text


# ============================================
# MULTI-VIEW DATASET CREATION
# ============================================

def create_multiview_dataset(df_meta, config):
    """
    Group images by study_id to create multi-view samples.
    
    Args:
        df_meta: Metadata dataframe with image records
        config: Multi-view configuration dictionary
        
    Returns:
        DataFrame with one row per study (multi-view samples)
    """
    print("\n" + "=" * 80)
    print("CREATING MULTI-VIEW DATASET")
    print("=" * 80)

    print(f"\nInput: {len(df_meta):,} image records")
    print(f"Unique studies in metadata: {df_meta['study_id'].nunique():,}")

    # Group by study_id
    print("\nGrouping images by study_id...")
    grouped = df_meta.groupby('study_id')

    multiview_samples = []
    
    # Statistics tracking
    stats = {
        'single_view': 0,
        'dual_view': 0,
        'triple_view': 0,
        'quad_plus_view': 0,
        'skipped_min_views': 0,
        'skipped_single_view': 0,
        'skipped_no_pa': 0,
        'truncated_max_views': 0
    }

    for study_id, group in tqdm(grouped, desc="Processing studies"):
        n_views = len(group)

        # Filter 1: Minimum views check
        if n_views < config['min_views_per_study']:
            stats['skipped_min_views'] += 1
            continue

        # Filter 2: Single view check (if not allowed)
        if not config['allow_single_view'] and n_views == 1:
            stats['skipped_single_view'] += 1
            continue

        # Filter 3: Require PA view (if configured)
        if config['require_pa'] and 'PA' not in group['ViewPosition'].values:
            stats['skipped_no_pa'] += 1
            continue

        # Filter 4: Truncate to max views if needed
        if n_views > config['max_views_per_study']:
            stats['truncated_max_views'] += 1
            view_order = {view: i for i, view in enumerate(config['view_preferences'])}
            group = group.copy()
            group['view_priority'] = group['ViewPosition'].map(lambda x: view_order.get(x, 999))
            group = group.sort_values('view_priority').head(config['max_views_per_study'])
            n_views = len(group)

        # Track view count distribution
        if n_views == 1:
            stats['single_view'] += 1
        elif n_views == 2:
            stats['dual_view'] += 1
        elif n_views == 3:
            stats['triple_view'] += 1
        else:
            stats['quad_plus_view'] += 1

        # Sort views by preference order for consistency
        view_order = {view: i for i, view in enumerate(config['view_preferences'])}
        group = group.copy()
        group['view_priority'] = group['ViewPosition'].map(lambda x: view_order.get(x, 999))
        group = group.sort_values('view_priority')

        # Extract sample information
        subject_id = group.iloc[0]['subject_id']

        # Collect image paths
        image_paths = [
            get_jpg_path(row['subject_id'], row['study_id'], row['dicom_id'])
            for _, row in group.iterrows()
        ]

        # Collect view positions - FIX: Convert NaN to "OTHER" (maintains alignment!)
        view_positions_raw = group['ViewPosition'].tolist()
        view_positions = [
            str(v) if pd.notna(v) else "OTHER"
            for v in view_positions_raw
        ]

        # Collect DICOM IDs
        dicom_ids = group['dicom_id'].tolist()

        # Create view combination label (for analysis, exclude OTHER)
        valid_positions = [v for v in view_positions if v != "OTHER"]
        if len(valid_positions) == 0:
            view_combination = "OTHER"
        else:
            view_combination = '+'.join(sorted(set(valid_positions)))

        # Get additional metadata
        patient_orientation = group.iloc[0].get('PatientOrientationCodeSequence_CodeMeaning', 'Unknown')
        procedure_description = group.iloc[0].get('PerformedProcedureStepDescription', 'Unknown')

        # Create sample record
        multiview_samples.append({
            'study_id': study_id,
            'subject_id': subject_id,
            'n_views': n_views,
            'view_combination': view_combination,
            'view_positions': view_positions,      # List aligned with image_paths
            'dicom_ids': dicom_ids,                # List aligned with image_paths
            'image_paths': image_paths,            # List of image paths
            'patient_orientation': patient_orientation,
            'procedure_description': procedure_description
        })

    df_multiview = pd.DataFrame(multiview_samples)

    # Print statistics
    print("\n" + "-" * 40)
    print("GROUPING STATISTICS")
    print("-" * 40)
    print(f"Total studies created: {len(df_multiview):,}")
    print(f"\nBy view count:")
    print(f"  1-view (single): {stats['single_view']:,}")
    print(f"  2-view (dual):   {stats['dual_view']:,}")
    print(f"  3-view (triple): {stats['triple_view']:,}")
    if stats['quad_plus_view'] > 0:
        print(f"  4+ views:        {stats['quad_plus_view']:,}")
    
    print(f"\nFiltered out:")
    if stats['skipped_min_views'] > 0:
        print(f"  Below min views: {stats['skipped_min_views']:,}")
    if stats['skipped_single_view'] > 0:
        print(f"  Single view (not allowed): {stats['skipped_single_view']:,}")
    if stats['skipped_no_pa'] > 0:
        print(f"  Missing PA view: {stats['skipped_no_pa']:,}")
    if stats['truncated_max_views'] > 0:
        print(f"  Truncated to max: {stats['truncated_max_views']:,}")

    return df_multiview


# ============================================
# REPORT ALIGNMENT
# ============================================

def align_multiview_reports(df_multiview):
    """
    Align multi-view studies with their reports and verify file existence.
    
    Args:
        df_multiview: DataFrame with multi-view samples
        
    Returns:
        DataFrame with aligned samples including report text
    """
    print("\n" + "=" * 80)
    print("ALIGNING WITH REPORTS")
    print("=" * 80)

    aligned_samples = []
    
    # Error tracking
    errors = {
        'missing_reports': 0,
        'empty_reports': 0,
        'missing_images': 0,
        'processing_errors': 0
    }

    print(f"\nProcessing {len(df_multiview):,} studies...")

    for idx, row in tqdm(df_multiview.iterrows(), total=len(df_multiview)):
        study_id = row['study_id']
        subject_id = row['subject_id']
        image_paths = row['image_paths']

        # Verify all images exist
        missing_images = [p for p in image_paths if not os.path.exists(p)]
        if missing_images:
            errors['missing_images'] += 1
            continue

        # Get report
        report_path = get_report_path(subject_id, study_id)
        
        if not os.path.exists(report_path):
            errors['missing_reports'] += 1
            continue

        try:
            with open(report_path, 'r', encoding='utf-8', errors='ignore') as f:
                report_text = f.read()

            # Clean report
            cleaned_report = format_clinical_note(report_text)

            if not cleaned_report or len(cleaned_report.strip()) < 10:
                errors['empty_reports'] += 1
                continue

            # Create aligned sample
            aligned_samples.append({
                'study_id': study_id,
                'subject_id': subject_id,
                'n_views': row['n_views'],
                'view_combination': row['view_combination'],
                'view_positions': row['view_positions'],
                'dicom_ids': row['dicom_ids'],
                'image_paths': image_paths,
                'patient_orientation': row['patient_orientation'],
                'procedure_description': row['procedure_description'],
                'report_path': report_path,
                'report_text': cleaned_report,
                'report_length': len(cleaned_report)
            })

        except Exception as e:
            errors['processing_errors'] += 1
            continue

    df_aligned = pd.DataFrame(aligned_samples)

    # Print statistics
    print("\n" + "-" * 40)
    print("ALIGNMENT STATISTICS")
    print("-" * 40)
    print(f"Successfully aligned: {len(df_aligned):,}")
    print(f"Success rate: {len(df_aligned)/len(df_multiview)*100:.1f}%")
    print(f"\nErrors:")
    print(f"  Missing reports: {errors['missing_reports']:,}")
    print(f"  Empty reports:   {errors['empty_reports']:,}")
    print(f"  Missing images:  {errors['missing_images']:,}")
    print(f"  Other errors:    {errors['processing_errors']:,}")

    if len(df_aligned) > 0:
        print(f"\nReport length stats:")
        print(f"  Mean:   {df_aligned['report_length'].mean():.0f} chars")
        print(f"  Median: {df_aligned['report_length'].median():.0f} chars")
        print(f"  Min:    {df_aligned['report_length'].min():.0f} chars")
        print(f"  Max:    {df_aligned['report_length'].max():.0f} chars")

    return df_aligned


# ============================================
# DATA SPLITTING
# ============================================

def split_dataset(df, train_ratio, val_ratio, test_ratio, seed):
    """
    Split dataset into train/val/test with no leakage.
    
    Args:
        df: DataFrame to split
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
        
    Returns:
        Tuple of (df_train, df_val, df_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # Shuffle deterministically
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Calculate split indices
    n_total = len(df_shuffled)
    train_end = int(train_ratio * n_total)
    val_end = train_end + int(val_ratio * n_total)

    # Split
    df_train = df_shuffled.iloc[:train_end].reset_index(drop=True)
    df_val = df_shuffled.iloc[train_end:val_end].reset_index(drop=True)
    df_test = df_shuffled.iloc[val_end:].reset_index(drop=True)

    return df_train, df_val, df_test


def verify_no_leakage(df_train, df_val, df_test):
    """Verify no study_id overlap between splits."""
    train_ids = set(df_train['study_id'])
    val_ids = set(df_val['study_id'])
    test_ids = set(df_test['study_id'])

    assert train_ids.isdisjoint(val_ids), "Train-Val leakage detected!"
    assert train_ids.isdisjoint(test_ids), "Train-Test leakage detected!"
    assert val_ids.isdisjoint(test_ids), "Val-Test leakage detected!"
    
    print("✓ No data leakage between splits")


# ============================================
# MAIN PIPELINE
# ============================================

def main():
    print("\n" + "=" * 80)
    print("MULTI-VIEW MIMIC-CXR PREPROCESSING PIPELINE")
    print("=" * 80)

    # Step 1: Load metadata
    print("\n[1/5] Loading metadata...")
    if not os.path.exists(META_PATH):
        print(f"ERROR: Metadata file not found: {META_PATH}")
        print("Please update META_PATH in the configuration section.")
        return
        
    df_meta = pd.read_csv(META_PATH)
    print(f"Loaded {len(df_meta):,} image records")
    print(f"Unique subjects: {df_meta['subject_id'].nunique():,}")
    print(f"Unique studies: {df_meta['study_id'].nunique():,}")

    # Step 2: Create multi-view dataset
    print("\n[2/5] Creating multi-view dataset...")
    df_multiview = create_multiview_dataset(df_meta, MULTIVIEW_CONFIG)

    if len(df_multiview) == 0:
        print("ERROR: No multi-view samples created. Check your configuration.")
        return

    # Step 3: Align with reports
    print("\n[3/5] Aligning with reports...")
    df_aligned = align_multiview_reports(df_multiview)

    if len(df_aligned) == 0:
        print("ERROR: No samples aligned. Check your paths.")
        return

    # Step 4: Split dataset
    print("\n[4/5] Splitting dataset...")
    df_train, df_val, df_test = split_dataset(
        df_aligned, 
        TRAIN_RATIO, 
        VAL_RATIO, 
        TEST_RATIO, 
        RANDOM_SEED
    )
    
    verify_no_leakage(df_train, df_val, df_test)

    # Step 5: Save files
    print("\n[5/5] Saving files...")
    
    # Save CSVs
    train_csv = os.path.join(OUTPUT_DIR, "multiview_train.csv")
    val_csv = os.path.join(OUTPUT_DIR, "multiview_val.csv")
    test_csv = os.path.join(OUTPUT_DIR, "multiview_test.csv")
    
    df_train.to_csv(train_csv, index=False, encoding='utf-8')
    df_val.to_csv(val_csv, index=False, encoding='utf-8')
    df_test.to_csv(test_csv, index=False, encoding='utf-8')
    
    print(f"Saved: {train_csv}")
    print(f"Saved: {val_csv}")
    print(f"Saved: {test_csv}")

    # Save Parquet (optional)
    if SAVE_PARQUET:
        df_train.to_parquet(os.path.join(OUTPUT_DIR, "multiview_train.parquet"), index=False)
        df_val.to_parquet(os.path.join(OUTPUT_DIR, "multiview_val.parquet"), index=False)
        df_test.to_parquet(os.path.join(OUTPUT_DIR, "multiview_test.parquet"), index=False)
        print("Saved: Parquet files")

    # Final summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    
    n_total = len(df_aligned)
    print(f"\nDataset Summary:")
    print(f"  Total samples: {n_total:,}")
    print(f"  Train: {len(df_train):,} ({len(df_train)/n_total:.1%})")
    print(f"  Val:   {len(df_val):,} ({len(df_val)/n_total:.1%})")
    print(f"  Test:  {len(df_test):,} ({len(df_test)/n_total:.1%})")
    
    print(f"\nView Distribution (Train):")
    view_counts = df_train['n_views'].value_counts().sort_index()
    for n_views, count in view_counts.items():
        print(f"  {n_views}-view: {count:,} ({count/len(df_train)*100:.1f}%)")
    
    print(f"\nTop View Combinations (Train):")
    top_combos = df_train['view_combination'].value_counts().head(5)
    for combo, count in top_combos.items():
        print(f"  {combo}: {count:,}")

    print("\n" + "=" * 80)
    print("READY FOR TRAINING")
    print("=" * 80)


if __name__ == "__main__":
    main()