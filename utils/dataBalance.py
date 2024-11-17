import sys
import os
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from collections import Counter

# Load the training dataset CSV
# print("CSV = TEST")
# train_data = pd.read_csv('../data/custom_sign_language_test.csv')

# print("CSV = TRAIN")
# train_data = pd.read_csv('../data/custom_sign_language_train.csv')

print("CSV = VAL")
train_data = pd.read_csv('../data/custom_sign_language_val.csv')

# Count occurrences of each class in the 'label' column
class_counts = Counter(train_data['label'])

# Sort the class counts in ascending order by class label
sorted_class_counts = dict(sorted(class_counts.items()))

# Display the number of samples per class in ascending order
for label, count in sorted_class_counts.items():
    print(f"Class {label}: {count} samples")
