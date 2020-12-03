from sklearn.ensemble import GradientBoostingRegressor
from datetime import timedelta


class BoostedTree:
    def __init__(self, **kwargs):
        # self.hypers = kwargs.get('hypers', {})
        self.trained_model = {}
        self.x_train = kwargs['x_train']
        self.x_test = kwargs['x_test']
        self.y_train = kwargs['y_train']
        self.permnos_test = kwargs['permnos_test']

    def fit(self):
        model = self.__get_model(0.5)
        model.fit(self.x_train, self.y_train, epochs=3, batch_size=1024)

    def __get_model(self, quantile):
        return GradientBoostingRegressor(loss="ls", alpha=quantile, n_estimators=100)

    def __tilted_loss(self, q, y, f):
        e = (y - f)
        return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)
