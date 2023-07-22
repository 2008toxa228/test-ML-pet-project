
import pandas as pd
import numpy as np
import dill as dill
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_validate


def check_overfitting(model, X_train, y_train, X_test, y_test, metric_fun):
    """
    Checks model for overfitting.

    Parameters:
       model (RegressorMixin): Model to be checked.
       X_train (DataFrame): Input data for model training.
       y_train (DataFrame): Target data for model training.
       X_test (DataFrame): Input data for model testing.
       y_test (DataFrame): Target data for model testing.
       metric_fun (function): Metric function.

    Returns:
        Difference between metric in train and test data as a percentage divided by 100.
    """

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    mae_train = metric_fun(y_train, y_pred_train)
    mae_test = metric_fun(y_test, y_pred_test)
    delta = round((abs(mae_train - mae_test)/mae_train), 3)

    return delta


def check_models(models, X, y, scoring, n_folds=5):
    """
    Calculating metrics for given models.

    Parameters:
       models (dict): Dictionary where the keys are the model names and the values are the estimators.
       X (DataFrame): Input data for models training.
       y (DataFrame): Target data for models training.
       scoring (dict): Dictionary where the keys are the metric names and the values are the metric scores .
       n_folds (DataFrame): Number of folds. Must be at least 2.

    Returns:
        Dataframe with name and scoring data for each model, sorted by.
    """

    models_results = []
    for model in models:
        scores = cross_validate(
            models[model], X, y, cv=n_folds, scoring=scoring)

        result = dict()
        result['model'] = model
        for score in scores:
            result[score] = {
                'mean': round(scores[score].mean(), 3),
                'max': round(scores[score].max(), 3)
            }

        models_results.append(result)

    results_df = pd.DataFrame(models_results).drop(
        ['fit_time', 'score_time'], axis=1)
    return results_df


def get_metrics(model, X_train, y_train, X_test, y_test):
    """
    Calculates metrics for model.

    Parameters:
       model (RegressorMixin): Model metrics calculation.
       X_train (DataFrame): Input data for model training.
       y_train (DataFrame): Target data for model training.
       X_test (DataFrame): Input data for model testing.
       y_test (DataFrame): Target data for model testing.

    Returns:
        Dictionary with metrics for model.
    """

    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    model_result = dict()
    model_result['R2'] = r2_score(y_test, predicted)
    model_result['explained_variance'] = explained_variance_score(
        y_test, predicted)
    model_result['MSE'] = mean_squared_error(y_test, predicted)
    model_result['MAE'] = mean_absolute_error(y_test, predicted)
    return model_result


def get_fold(n_fold, cross_calidator, X, y):
    """
    Calculates metrics for model.

    Parameters:
       n_fold (int): Fold number in cross validator.
       cross_calidator (BaseCrossValidator): Cross validator to use for splitting data.
       X (DataFrame): Input data.
       y (DataFrame): Target data.

    Returns:
        Dictionary with metrics for model.
    """

    train_index, test_index = list(cross_calidator.split(X, y))[n_fold]
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    return X_train, y_train, X_test, y_test
