# Import necessary libraries
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import joblib

# Load the dataset
file_path = 'main_dataset.xlsx'  # Make sure the file path is correct
df = pd.read_excel(file_path)

# Extract features (X) and target variables (Y)
X = df[['Number of Students']]
Y_main_meter = df['Main Meter']
Y_resource_centre = df['Resource Centre']

# Split the data into training and testing sets
X_train, X_test, Y_main_meter_train, Y_main_meter_test, Y_resource_centre_train, Y_resource_centre_test = train_test_split(
    X, Y_main_meter, Y_resource_centre, test_size=0.2, random_state=42
)

# Decision Tree Regression for Main Meter
model_main_meter_dt = DecisionTreeRegressor(random_state=42)
model_main_meter_dt.fit(X_train, Y_main_meter_train)

# Decision Tree Regression for Resource Centre
model_resource_centre_dt = DecisionTreeRegressor(random_state=42)
model_resource_centre_dt.fit(X_train, Y_resource_centre_train)

# Make predictions
Y_main_meter_pred_dt = model_main_meter_dt.predict(X_test)
Y_resource_centre_pred_dt = model_resource_centre_dt.predict(X_test)

# Calculate metrics
mse_main_meter_dt = mean_squared_error(Y_main_meter_test, Y_main_meter_pred_dt)
mae_main_meter_dt = mean_absolute_error(Y_main_meter_test, Y_main_meter_pred_dt)
r2_main_meter_dt = r2_score(Y_main_meter_test, Y_main_meter_pred_dt)

mse_resource_centre_dt = mean_squared_error(Y_resource_centre_test, Y_resource_centre_pred_dt)
mae_resource_centre_dt = mean_absolute_error(Y_resource_centre_test, Y_resource_centre_pred_dt)
r2_resource_centre_dt = r2_score(Y_resource_centre_test, Y_resource_centre_pred_dt)

# Print metrics
print("Decision Tree Main Meter Metrics:")
print("Mean Squared Error:", mse_main_meter_dt)
print("Mean Absolute Error:", mae_main_meter_dt)
print("R-squared:", r2_main_meter_dt)

print("\nDecision Tree Resource Centre Metrics:")
print("Mean Squared Error:", mse_resource_centre_dt)
print("Mean Absolute Error:", mae_resource_centre_dt)
print("R-squared:", r2_resource_centre_dt)

# Save the pre-trained models if needed
joblib.dump(model_main_meter_dt, 'model_main_meter_dt.pkl')
joblib.dump(model_resource_centre_dt, 'model_resource_centre_dt.pkl')

# Plotting actual vs predicted for Main Meter
plt.scatter(X_test, Y_main_meter_test, color='black', label='Actual')
plt.scatter(X_test, Y_main_meter_pred_dt, color='blue', label='Predicted')
plt.title('DT Main Meter: Actual vs Predicted')
plt.xlabel('Number of Students')
plt.ylabel('Main Meter')
plt.legend()
plt.show()

# Plotting actual vs predicted for Resource Centre
plt.scatter(X_test, Y_resource_centre_test, color='black', label='Actual')
plt.scatter(X_test, Y_resource_centre_pred_dt, color='red', label='Predicted')
plt.title('DT Resource Centre: Actual vs Predicted')
plt.xlabel('Number of Students')
plt.ylabel('Resource Centre')
plt.legend()
plt.show()