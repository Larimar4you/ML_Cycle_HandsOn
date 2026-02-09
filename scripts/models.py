from sklearn.linear_model import LinearRegression, Lasso, Ridge


def train_model(X_train, y_train, model_type="linear", alpha=1.0):
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "lasso":
        model = Lasso(alpha=alpha)
    elif model_type == "ridge":
        model = Ridge(alpha=alpha)
    else:
        raise ValueError("Unknown model type")

    model.fit(X_train, y_train)
    return model
