# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Iris dataset and separate features (X) and target labels (y).
2. Split the dataset into training and testing sets.
3. Apply feature scaling using StandardScaler to normalize the data.
4. Train the SGD Classifier model using the training data.
5. Predict the species for test data and evaluate the model using accuracy and classification report.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: MOHAMMED AFSAL S
RegisterNumber: 212225040247 
*/
```
```
# Import required libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# Step 1: Load Iris Dataset
# -------------------------------

iris = load_iris()

X = iris.data
y = iris.target

# Convert to DataFrame for display
df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = y

print("Sample Dataset:\n")
print(df.head())

# -------------------------------
# Step 2: Split Dataset
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------
# Step 3: Create Pipeline (Scaling + SGD)
# -------------------------------

model = make_pipeline(
    StandardScaler(),
    SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
)

# Train model
model.fit(X_train, y_train)

# -------------------------------
# Step 4: Predictions
# -------------------------------

y_pred = model.predict(X_test)

# -------------------------------
# Step 5: Evaluation
# -------------------------------

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
```

## Output:
<img width="477" height="498" alt="image" src="https://github.com/user-attachments/assets/68ee2ede-700c-420e-9c6c-31e44a4927ee" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
