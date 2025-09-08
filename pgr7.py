from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



iris = load_iris()
X=iris.data  
y=iris.target 


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,         
    min_samples_split=4,
    min_samples_leaf=2,  
    max_features='sqrt',
    random_state=42 ,   
     ccp_alpha=0.01  
)
 

model.fit(X_train,y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


plt.figure(figsize=(12,8))
plot_tree(model,filled=True,feature_names=iris.feature_names,
          class_names=iris.target_names, rounded=True)
plt.show()