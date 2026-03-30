import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def parse_filename(filename):
    """
    Parse UTKFace filename format: age_gender_race_date.jpg
    Returns tuple (age, gender, race) or None if pattern doesn't match
    """
    pattern = re.compile(r"^(\d+)_(\d+)_(\d+)_([0-9]+)\.jpg(?:\.chip)?\.jpg$")
    match = pattern.match(filename)
    
    if not match:
        return None
    
    age = int(match.group(1))
    gender = int(match.group(2))
    race = int(match.group(3))
    
    return age, gender, race

def load_data(data_dir, age_min=0, age_max=116):
    """
    Load UTKFace dataset from directory
    Returns DataFrame with columns: path, age, gender, race
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.jpg"))
    
    rows = []
    bad = 0
    
    for fp in files:
        parsed = parse_filename(fp.name)
        if parsed is None:
            bad += 1
            continue
        
        age, gender, race = parsed
        rows.append((str(fp), age, gender, race))
    
    df = pd.DataFrame(rows, columns=["path", "age", "gender", "race"])
    
    # Filter valid ages
    df = df[(df["age"] >= age_min) & (df["age"] <= age_max)].copy()
    
    return df, bad

def create_age_bins(df, bins=[0,3,13,20,30,40,50,60,70,80,117],
                    labels=["0-2","3-12","13-19","20-29","30-39","40-49","50-59","60-69","70-79","80+"]):
    """Add age_bin column to dataframe"""
    df["age_bin"] = pd.cut(df["age"], bins=bins, right=False, labels=labels)
    return df, labels

def train_val_split(df, test_size=0.2, random_state=42, stratify_col="age_bin"):
    """Split data with stratification by age bin"""
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df[stratify_col]
    )
    return train_df, val_df
