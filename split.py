"""
Part of Phase 7: train/validation/test split
------------------------------------------------
WHY A PLAIN RANDOM SPLIT (not stratified, not time-based): this dataset
has no listing date at all -- only the car's manufacture year, which is
a FEATURE (age), not a timestamp of when the observation was collected.
There is no time axis to split on (unlike Day 2/11), and no severe class
imbalance requiring stratification (this is a regression problem, not
classification, so there's no class to stratify on in the first place --
a plain random split is the correct, unforced default here).
"""

from sklearn.model_selection import train_test_split


def split_data(X, y, test_size=0.15, val_size=0.15, random_state=42):
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_relative_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test
