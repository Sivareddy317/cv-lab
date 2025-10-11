import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# ------------------------
# 1. Generate dataset with more features
# ------------------------
X, y = make_classification(
    n_samples=500,
    n_features=10,     # more features
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    random_state=42
)

# ------------------------
# 2. Apply PCA (reduce to 2D for visualization)
# ------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Explained variance ratio:", pca.explained_variance_ratio_)

# ------------------------
# 3. Train-test split on reduced features
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, random_state=42, stratify=y
)

# ------------------------
# 4. Train Logistic Regression
# ------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ------------------------
# 5. Evaluate
# ------------------------
y_pred = model.predict(X_test)
print("Accuracy after PCA:", accuracy_score(y_test, y_pred))

# ------------------------
# 6. Visualization
# ------------------------
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="coolwarm", edgecolors="k", s=60)
plt.title("PCA Projection of Dataset (2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Class")
plt.show()
