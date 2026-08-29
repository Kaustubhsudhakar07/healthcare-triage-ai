"""
Stratified Train/Test Dataset Splitting
AI-Assisted Pre-Hospital Patient Criticality Prediction System

Splits raw dataset into stratified train (80%) and test (20%) partitions
based on operational urgency category.
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def split_and_save_data(
    input_path: str = "data/raw/patient_criticality_data.csv",
    output_dir: str = "data/processed",
    test_size: float = 0.20,
    random_state: int = 42
) -> None:
    """
    Performs stratified train/test split and writes output files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading raw data from '{input_path}'...")
    df = pd.read_csv(input_path)
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["urgency_category"]
    )
    
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Train split saved to: {train_path} ({len(train_df):,} records)")
    print(f"Test split saved to:  {test_path} ({len(test_df):,} records)")
    
    print("\nTrain Category Proportions:")
    print(train_df["urgency_category"].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%"))
    print("\nTest Category Proportions:")
    print(test_df["urgency_category"].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%"))


def main():
    parser = argparse.ArgumentParser(description="Split raw dataset into train and test sets.")
    parser.add_argument("--input_path", type=str, default="data/raw/patient_criticality_data.csv")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    split_and_save_data(
        input_path=args.input_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.seed
    )


if __name__ == "__main__":
    main()
