import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

def get_conf_matrix(model, x, y, classes, n_splits=5):    
    """
    
    Parameters:
    -----------
    model : `sklearn` model
        The model to cross validate.
    x : `list`
        The training data.
    y : `list`
        The corresponding labels.
    classes : `list` of `str`
        The class labels.
    n_splits : `int`
        The number of splits for k-fold cross-validation.
    
    Returns:
    --------
    random_search_res : ``
        pass
    """  
    # Define a CV strategy
    cv = StratifiedKFold(n_splits=n_splits,
                         shuffle=True,
                         random_state=79)  
    # Cross-validation
    conf_matrices = np.zeros((4,4,n_splits))
    class_reports = np.zeros((4,4,n_splits))
    for i, (train_index, test_index) in enumerate(cv.split(x,y)):
        train_x, test_x = x[train_index], x[test_index]
        train_y, test_y = y[train_index], y[test_index]
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        # Get confusion matrix
        conf_matrices[:,:,i] = confusion_matrix(test_y, pred_y, normalize=None)
        # Get classification report
        report = classification_report(test_y, pred_y, target_names=classes, output_dict=True)
        class_reports[:,:,i] = pd.DataFrame.from_dict({key:report[key] for key in classes}).to_numpy()
        
    # Average confusion matrix
    aug_conf_matrices = copy.deepcopy(conf_matrices)
    for i in range(n_splits):
        conf_matrix = aug_conf_matrices[:,:,i]
#         # Normalise along rows and fill nans caused by 0/0 to 0
#         conf_matrix = conf_matrix/np.sum(conf_matrix, axis=1)[:,None]
#         np.nan_to_num(conf_matrix, copy=False, nan=0)
        # Fill the diagonal of the confusion matrix with 0
        np.fill_diagonal(conf_matrix , 0)
        conf_matrix = conf_matrix/np.sum(conf_matrix)
        aug_conf_matrices[:,:,i] = conf_matrix 
    # Get means and std
    mean_conf_matrix = np.mean(aug_conf_matrices,axis=2)*100 
    std_conf_matrix = np.std(aug_conf_matrices,axis=2)*100  
    
    # Average classification report
    mean_class_reports = np.mean(class_reports, axis=2).round(4)
    std_class_reports = np.std(class_reports, axis=2).round(4)
    mean_class_reports = np.char.array(mean_class_reports.astype(str))
    std_class_reports = np.char.array(std_class_reports.astype(str))
    class_reports = mean_class_reports+np.full(mean_class_reports.shape, "\pm")+std_class_reports
    
    return mean_conf_matrix.round(2), std_conf_matrix.round(2), class_reports

def make_row(table, row, model_name):
    """
    Parameters:
    -----------
    table : `collections.defaultdict`
        The dictionary to fill.
    row : `pandas.DataFrame`
        The dataframe to get the data from.
    model_name : `str`
        The model name.
    
    Returns:
    --------
    table : `collections.defaultdict`
        The filled dictionary.
    """
    labels = ['Accuracy (%)','Precision', 'Recall', 'F$1$-Score', ' MCC']
    headers = list(row.keys())[1:]
    table['Model'].append(model_name)
    for i, label in enumerate(labels):
        table[label].append(str(row[headers[i*2]])+"±"+str(row[headers[i*2+1]]))
    return table