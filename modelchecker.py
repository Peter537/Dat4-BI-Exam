from sklearn.metrics import mean_squared_error
from math import sqrt

class ModelChecker:
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def run(self):
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        return self

    def show_accuracy(self):
        score = self.model.score(self.X_test, self.y_test)
        print(f"R2 accuracy score: {score}")

        mse = mean_squared_error(self.y_test, self.y_pred)
        print(f"Mean squared error: {mse}")

        rmse = sqrt(mse)
        print(f"Root mean squared error: {rmse}")

        return score, mse, rmse