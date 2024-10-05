import pandas as pd
from collections import Counter

# Load the training dataset CSV
train_data = pd.read_csv('../data/sign_mnist_train.csv')

# Count occurrences of each class in the 'label' column
class_counts = Counter(train_data['label'])

# Display the number of samples per class
for label, count in class_counts.items():
    print(f"Class {label}: {count} samples")
