import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler_std = StandardScaler()
scaler_minmax = MinMaxScaler()

X_std = scaler_std.fit_transform(X)
X_minmax = scaler_minmax.fit_transform(X)

plt.figure(figsize=(10, 5))
plt.plot(X[:, 0], label="Original")
plt.plot(X_std[:, 0], label="StandardScaler")
plt.plot(X_minmax[:, 0], label="MinMaxScaler")
plt.legend()
plt.title("Сравнение нормализации")
plt.show()
