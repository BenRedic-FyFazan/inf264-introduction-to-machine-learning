import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score


def valid_eval_method(eval_method):
    """
    Validates the evaluation method.
    """
    valid_methods = {'hold-out', 'cross-validation'}
    if eval_method not in valid_methods:
        raise ValueError('Invalid eval_method. Supported evaluation methods are {}'.format(valid_methods))
    return eval_method

def valid_classifier(clf):
    """
    Validates that the classifier has the necessary methods.
    """
    required_methods = {'fit', 'predict'}
    for method in required_methods:
        if not hasattr(clf, method):
            raise ValueError('Invalid classifier. Classifier must have method {}'.format(method))
    return clf

def valid_scoring_method(method):
    """
    Validates the scoring method.
    """
    required_methods = {'accuracy', 'f1'}
    if method not in required_methods:
        raise ValueError('Invalid scoring method. Supported scoring methods are {}'.format(required_methods))
    return method

def hyperparameter_combinations(hyperparameter_grid):
    """
    Generates all possible combinations of hyperparameters.
    """
    keys = hyperparameter_grid.keys()
    values = hyperparameter_grid.values()
    product_combinations = product(*values)
    hyperparameter_dicts = [dict(zip(keys, combination)) for combination in product_combinations]

    return hyperparameter_dicts

class ModelEvaluation:
    """
    Represents the evaluation result of a model with specific hyperparameters.

    Attributes:
    -----------
    hyperparameters : dict or None
        The hyperparameters used for the model.
    score : float
        The evaluation score of the model.
    """
    def __init__(self, hyperparameters=None, score=-np.inf ):
        self.hyperparameters = hyperparameters
        self.score = score


class ClassifierEval:
    """
    Evaluates classifiers using specified evaluation methods and hyperparameter grids.

    Parameters:
    -----------
    clf : object, optional
        The classifier to evaluate. Must implement 'fit' and 'predict' methods. Defaults to None.
    hyperparameter_grid : dict, optional
        A dictionary specifying hyperparameters and their possible values. Defaults to None.
    x : array-like, optional
        Feature data. Defaults to None.
    y : array-like, optional
        Target labels. Defaults to None.
    eval_method : str, optional
        Evaluation method to use ('hold-out' or 'cross-validation'). Defaults to 'hold-out'.
    test_size : float, optional
        Proportion of the dataset to include in the test split. Relevant for 'hold-out' method. Defaults to 0.3.
    cv_folds : int, optional
        Number of folds for cross-validation. Relevant for 'cross-validation' method. Defaults to 5.
    scoring_method : str, optional
        Scoring metric to use ('accuracy' or 'f1'). Defaults to 'accuracy'.
    random_state : int or None, optional
        Seed for random number generator. Defaults to None.
    """
    def __init__(self,
                 clf=None,
                 hyperparameter_grid=None,
                 x=None,
                 y=None,
                 eval_method='hold-out',
                 test_size = 0.2,
                 cv_folds = 5,
                 scoring_method = 'accuracy',
                 random_state=None):
        self.clf = valid_classifier(clf)
        self.hyperparam_grid = hyperparameter_grid
        self.hyperparam_combinations = hyperparameter_combinations(hyperparameter_grid)
        self.x = x
        self.y = y
        self.eval_method = valid_eval_method(eval_method)
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.scoring_method = valid_scoring_method(scoring_method)
        self.evaluated_models = []
        self.best_model = ModelEvaluation()

        if random_state is not None:
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = np.random.RandomState()

    def evaluate(self):
        """
        Evaluates the classifier using the specified evaluation method.
        """
        if self.eval_method == 'cross-validation':
            self._cross_validation()
        else:
            self._hold_out()

    def _hold_out(self):
        """
        Evaluates the classifier using the hold-out method.
        Splits the data into training and testing sets and evaluates each hyperparameter combination.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            self.x,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        for hyperparams in self.hyperparam_combinations:
            clf = self.clf(**hyperparams)
            clf.fit(x_train, y_train)
            predictions = clf.predict(x_test)
            score = self._score(y_test, predictions)
            res_model = ModelEvaluation(
                hyperparameters=hyperparams,
                score=score,
            )
            self.evaluated_models.append(res_model)
            self._update_best_model(res_model)


    def _cross_validation(self):
        """
        Evaluates the classifier using cross-validation.
        Performs K-Fold cross-validation for each hyperparameter combination and averages the scores.
        """
        kfolds = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        for hyperparams in self.hyperparam_combinations:
            scores = []
            for fold, (train_index, test_index) in enumerate(kfolds.split(self.x), 1):
                x_train, x_test = self.x[train_index], self.x[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]

                clf = self.clf(**hyperparams)
                clf.fit(x_train, y_train)
                predictions = clf.predict(x_test)
                scores.append(self._score(y_test, predictions))

            mean_score = np.mean(scores)
            res_model = ModelEvaluation(
                hyperparameters=hyperparams,
                score = mean_score,
            )
            self.evaluated_models.append(res_model)
            self._update_best_model(res_model)


    def _score(self, y_test, predictions):
        """
        Computes the evaluation score based on the selected scoring method.
        """
        if self.scoring_method == 'f1':
            return f1_score(y_test, predictions, average='weighted')
        else:
            return accuracy_score(y_test, predictions)

    def _update_best_model(self, classifier_model):
        """
        Updates the best model if the current model has a higher score.
        """
        if classifier_model.score > self.best_model.score:
            self.best_model = classifier_model

    def get_best_hyperparameters(self):
        """
        Retrieves the hyperparameters of the best model.
        """
        return self.best_model.hyperparameters

    def get_evaluated_models(self, sort_by_score=False, ascending=False):
        """
        Retrieves all evaluated models with their hyperparameters and scores.
        """
        data = []
        for model in self.evaluated_models:
            row = model.hyperparameters.copy()
            row[f"{self.scoring_method}_score"] = model.score
            data.append(row)

        if sort_by_score is True:
            data.sort(key=lambda x: x[f"{self.scoring_method}_score"], reverse = not ascending)

        return data

    def get_best_score(self):
        """
        Retrieves the best evaluation score achieved.
        """
        return self.best_model.score
