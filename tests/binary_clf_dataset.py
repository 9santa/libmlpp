from sklearn.datasets import make_classification
import csv

numFeatures = 10

X, y = make_classification(
    n_samples=50000,
    n_features=numFeatures,
    n_informative=numFeatures,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=1,
    class_sep=2.0,
    random_state=42,
)

y = [1 if label == 1 else -1 for label in y]

with open("binary_dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)

    header = [f"x{i + 1}" for i in range(numFeatures)] + ["label"]
    writer.writerow(header)

    for features, label in zip(X, y):
        writer.writerow(list(features) + [label])
