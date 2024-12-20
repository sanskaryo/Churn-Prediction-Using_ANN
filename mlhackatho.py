from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Select features and target variable
X = df.drop(columns=["Fuel_Efficiency"])
y = df["Fuel_Efficiency"]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(include=["number"]).columns

# One-hot encode categorical columns
encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
X_categorical = encoder.fit_transform(X[categorical_cols])

# Normalize numerical columns
scaler = StandardScaler()
X_numerical = scaler.fit_transform(X[numerical_cols])

# Combine preprocessed data
X_processed = np.hstack((X_categorical, X_numerical))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mse, r2
