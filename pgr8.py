import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------
# 1. Generate Synthetic Dataset
# ------------------------
X, y = make_classification(
    n_samples=500,       # number of samples
    n_features=2,        # number of features (for easy visualization)
    n_classes=2,         # binary classification
    n_informative=2,
    n_redundant=0,
    random_state=42
)

# Convert to DataFrame for clarity
df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
df['Label'] = y

# ------------------------
# 2. Train-Test Split
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ------------------------
# 3. Train Logistic Regression
# ------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ------------------------
# 4. Predictions
# ------------------------
y_pred = model.predict(X_test)

# ------------------------
# 5. Evaluation
# ------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------
# 6. Visualization (Decision Boundary)
# ------------------------
plt.figure(figsize=(8,6))
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='coolwarm', edgecolors='k', s=60, label="True Labels")

# Plot decision boundary
x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.legend()
plt.show()
