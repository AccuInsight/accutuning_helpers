import numpy as np


class EnsembleProxy(object):
    def __init__(self, objects_with_weight):
        self.models = [
            e[1]
            for e in objects_with_weight
        ]
        self.weights_ = [
            e[0]
            for e in objects_with_weight
        ]

    def predict(self, X):
        all_predictions = [
            model.predict(X.copy())  # predict_proba
            for model in self.models
        ]
        if len(all_predictions) > 1:
            return self.ensemble_predictions(all_predictions)
        else:
            return all_predictions[0]

    def predict_proba(self, X):
        # TODO: 모든 모델에 대해서 수행하도록; using np.average with weights
        prediction = self.models[0].predict_proba(X.copy())  # predict_proba
        return prediction

    def ensemble_predictions(self, predictions):
        predictions = np.asarray(predictions)

        if predictions.shape[0] == len(self.weights_):
            return np.average(predictions, axis=0, weights=self.weights_)

        # predictions do not include those of zero-weight models.
        elif predictions.shape[0] == np.count_nonzero(self.weights_):
            non_null_weights = [w for w in self.weights_ if w > 0]
            return np.average(predictions, axis=0, weights=non_null_weights)

        # If none of the above applies, then something must have gone wrong.
        else:
            raise ValueError("The dimensions of ensemble predictions"
                             " and ensemble weights do not match!")
