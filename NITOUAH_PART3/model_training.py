import pandas as pd
import pickle
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from assets_data_prep import prepare_data

# 1. Load the training dataset
df = pd.read_excel("train.xlsx")

# 2. Clean and preprocess the data using prepare_data
df = prepare_data(df, mode="train")

# 3. Split features and target
X = df.drop(columns=["price"])
y = df["price"]

# 4. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
    
with open("columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)
    
# 5. Train ElasticNet model
model = ElasticNet(alpha=1.0, l1_ratio=0.3, random_state=42)
model.fit(X_scaled, y)

# 6. Save the trained model
with open("trained_model.pkl", "wb") as f:
    pickle.dump(model, f)
