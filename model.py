import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# 1️⃣ Load the data
train_df = pd.read_csv('/Users/praveenan/Desktop/Kaggle competition/Dateset/train.csv')
test_df = pd.read_csv('/Users/praveenan/Desktop/Kaggle competition/Dateset/test.csv')
submission_df = pd.read_csv('/Users/praveenan/Desktop/Kaggle competition/Dateset/sample_submission.csv')

# 2️⃣ Clean 'running' (mileage) in both datasets
def clean_running(x):
    try:
        return int(str(x).replace('km', '').replace('KM', '').replace(' ', '').replace(',', '').strip())
    except:
        return 0

train_df['running'] = train_df['running'].apply(clean_running)
test_df['running'] = test_df['running'].apply(clean_running)

# 3️⃣ Remove price outliers (keep prices under $50,000)
train_df = train_df[train_df['price'] < 50000]

# 4️⃣ Add 'vehicle_age' feature
train_df['vehicle_age'] = 2025 - train_df['year']
test_df['vehicle_age'] = 2025 - test_df['year']

# 5️⃣ Encode categorical features safely
cat_cols = ['model', 'motor_type', 'wheel', 'color', 'type', 'status']
encoder = LabelEncoder()

for col in cat_cols:
    combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
    encoder.fit(combined)
    
    train_df[col] = encoder.transform(train_df[col].astype(str))
    test_df[col] = encoder.transform(test_df[col].astype(str))

# 6️⃣ Feature selection
features = ['year', 'vehicle_age', 'motor_volume', 'model', 'motor_type', 'running', 'wheel', 'type']
X = train_df[features]
y = train_df['price']

# 7️⃣ Train-test split for local evaluation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 8️⃣ Train improved Random Forest
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)

# 9️⃣ Evaluate the improved model
y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"🚀 Improved Mean Absolute Error: {mae:.2f}")
print(f"🚀 Improved R² Score: {r2:.2f}")

# 🔟 Predict on test set
X_test = test_df[features]
test_predictions = model.predict(X_test)

# 11️⃣ Create submission
submission_df['price'] = test_predictions
submission_df.to_csv('/Users/praveenan/Desktop/Kaggle competition/Dateset/Car_prediciton_update_final.csv', index=False)

print("✅ Improved submission file created: Car_prediciton_update_final.csv")
