import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ Загрузка данных
df = pd.read_csv("sales.csv")
print("Перші рядки даних:\n", df.head())

# 2️⃣ Очистка даних
df = df.dropna()  # видаляємо пропуски
for col in ["Price", "Advertising", "Stock", "Sales"]:
    df = df[df[col] >= 0]  # видаляємо від'ємні значення

# 3️⃣ Масштабування (для лінійної регресії)
X = df[["Price", "Advertising", "Stock"]]
y = df["Sales"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ EDA
sns.pairplot(df)
plt.show()

corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()

# 5️⃣ Розділяємо на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 6️⃣ Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("Linear Regression R²:", r2_score(y_test, y_pred_lr))

# 7️⃣ Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("Random Forest R²:", r2_score(y_test, y_pred_rf))

# Важливість ознак
feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
feature_importances.sort_values().plot(kind="barh")
plt.title("Feature Importance (Random Forest)")
plt.show()

# 8️⃣ Маленький висновок
print("\nМодель показала, які ознаки впливають на продажі найбільше:")
print(feature_importances)
