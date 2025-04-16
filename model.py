import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer,StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import boxcox
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, recall_score, cohen_kappa_score
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")

columns = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']

df = df.drop_duplicates().reset_index(inplace = False)
fitted_lambda_dict = {}
for col in columns:
  df[f"{col}_box"] = df[col] + 0.001
  df[f"{col}_box"], fitted_lambda = boxcox(df[f"{col}_box"])
  fitted_lambda_dict[col] = fitted_lambda

print("Training model......")

X = df.drop(columns=['Diabetes_binary'])
X = X.drop(columns=['index', 'BMI', 'GenHlth_box', 'MentHlth_box', 'PhysHlth_box', 'Age_box', 'Education_box', 'Income_box'])

y = df['Diabetes_binary']

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Over sampling
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

#Splitting
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Model
rf = RandomForestClassifier(n_estimators = 100, max_depth = None, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

r2 = r2_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cohen_k = cohen_kappa_score(y_test, y_pred)

model_details = {"Model": "Random Forest",
                 "Accuracy": accuracy,
                 "Recall": recall,
                 "Cohen Kappa Score": cohen_k,
                 "Fitted Lambda": fitted_lambda_dict}

# Save model and scaler
joblib.dump(rf, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")
joblib.dump(model_details, "model_details.pkl")

print("Model, scaler, and feature names saved successfully.")