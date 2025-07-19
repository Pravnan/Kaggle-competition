import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
train_df = pd.read_csv('/Users/praveenan/Desktop/Kaggle competition/Dateset/train.csv')
test_df = pd.read_csv('/Users/praveenan/Desktop/Kaggle competition/Dateset/test.csv')

# 1️⃣ Basic Information
print("Train Shape:", train_df.shape)
print("\nData Types:\n", train_df.dtypes)
print("\nMissing Values:\n", train_df.isnull().sum())

# 2️⃣ Summary Statistics
print("\nSummary Statistics:\n", train_df.describe())

# 3️⃣ Distributions of Numeric Columns
numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(train_df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# 4️⃣ Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_matrix = train_df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# 5️⃣ Optional: Boxplots for Outliers
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=train_df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()
