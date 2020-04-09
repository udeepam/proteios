from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer

def cross_validate(model, x, y, params, n_splits=5):
    """
    
    Parameters:
    -----------
    model : `sklearn` model
        The model to cross validate.
    x : `list`
        The training data.
    y : `list`
        The corresponding labels.
    params : `dict`
        The hyperparameters to cross-validate.
    n_splits : `int`
        The number of splits for k-fold cross-validation.
    
    Returns:
    --------
    random_search_res : ``
        pass
    """
    scoring = {"Accuracy": make_scorer(accuracy_score),
               "Precision": make_scorer(precision_score, average='macro'),
               "Recall": make_scorer(recall_score, average='macro'),
               "F1": make_scorer(f1_score, average='macro'),
               "MCC": make_scorer(matthews_corrcoef)}
    
    # Define a CV strategy
    cv = StratifiedKFold(n_splits=n_splits,
                         shuffle=True,
                         random_state=79)    

    # Define how we are going to fit our model parameters using the CV, hyperparameters and what evaluation metric
    grid_search = GridSearchCV(model,
                               param_grid=params,
                               verbose=3,
                               scoring=scoring,
                               n_jobs=1,
                               refit="Accuracy",
                               cv=cv)

    # Fit the model parameters using the defined search
    random_search_res = grid_search.fit(x, y)   
    return random_search_res