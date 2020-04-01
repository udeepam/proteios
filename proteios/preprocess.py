import numpy as np
import collections

from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from proteios.utils.numpy import shuffle_data

def modify_seq(data):
    """
   `X` : replace with any amino acid from list including `B` and `U`
   `B` : replace with either D or N
   `U` : replace with `C` 
    
    Parameters:
    -----------
    data : `list` of `Bio.SeqRecord.SeqRecord`
        List of sequence records.
    
    Returns:
    --------
    modified_data : `list` of `Bio.SeqRecord.SeqRecord`
        The data with the modifications.
    """
    amino_acids = ['A','B','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','U','V','W','Y']
    for example in data:
        tmp = list(str(example.seq))
        for i, char in enumerate(tmp):
            if char=='X':
                char = np.random.choice(amino_acids)
                tmp[i] = char
            if char=='B':
                if np.random.uniform() > 0.5:
                    tmp[i] = 'D'
                else:
                    tmp[i] = 'N'
            elif char=='U':
                tmp[i] = 'C'
        tmp = ''.join(tmp)
        example.seq = Seq(tmp) 
    return data

def remove_outlier_lengths(seq_record_data, protein_analysis_data, max_length=2000):
    """
    Parameters:
    -----------
    seq_record_data : `list` of `Bio.SeqRecord.SeqRecord`
        List of sequence records.    
    protein_analysis_data : `list` of `Bio.SeqUtils.ProtParam.ProteinAnalysis`
        List of sequence records.
    max_length : `int`
        The maximum length of the sequence.        
    
    Returns:
    --------
    modified_data : `list` of `Bio.SeqRecord.SeqRecord`
        The data with outlier lengths removed.
    """    
    modified_data = [seq_record_data[i] for i, example in enumerate(protein_analysis_data) if example.length < max_length]
    print("Data removed: ", len(seq_record_data)-len(modified_data))
    return modified_data

def preprocess(data, label2index=None, trim_outliers=True, max_length=2000):
    """
    Function that takes a dictionary of sequences and 
    preprocesses.
    Produces corresponding class index label.
    
    Parameters:
    -----------
    data : `dict` of `list` of `Bio.SeqRecord.SeqRecord` or `list` of `Bio.SeqRecord.SeqRecord`
        The dataset where each key is a different class.
    label2index : `dict`
        Dictionary mapping class label to index.
    trim_outliers : `Boolean`
        Whether to remove outlier data.
    max_length : `int`
        The maximum length of the sequence.
    
    Returns:
    --------
    proc_data : `list` of `Bio.SeqRecord.SeqRecord`
        Processed dataset.
    proc_labels : `list` of `ints`
        Corresponding labels.
    """
    if isinstance(data, list):
        data = {"tmp": data}
    # Preprocessing
    counts = collections.defaultdict(int)
    for i, (key, val) in enumerate(data.items()):
        print("Processing "+str(key))
        counts['before'] += len(val)
        # Preprocess the sequence to remove `X` and `B` and `U`
        val = modify_seq(val)
        # Convert Bio.SeqRecord.SeqRecord object to string for Bio.SeqUtils.ProtParam.ProteinAnalysis
        proto_val = [ProteinAnalysis(str(example.seq)) for example in val]
        # Remove outlier lengths
        if trim_outliers:
            val = remove_outlier_lengths(val, proto_val, max_length=max_length)
        data[key] = val
        counts['after'] += len(val)
    print("Total Before: ", counts['before'])
    print("Total After:  ", counts['after'])    
    
    if "tmp" in data:
        return data["tmp"]
    else:
        proc_data = collections.defaultdict(list)
        for i, (key, val) in enumerate(data.items()):
            # Get data
            proc_data["data"].extend(val)
            # Get corresponding labels
            proc_data["labels"].extend([label2index[key]]*len(val))  
        # Shuffle data
        return shuffle_data(proc_data["data"], proc_data["labels"])
       

def split_data(data, labels, train_size):
    """
    Splits data.
    
    Parameters:
    -----------
    data : `list` of `Bio.SeqRecord.SeqRecord`
        The dataset where each key is a different class.
    labels : `list` of `int`
        Corresponding labels.
    train_size : `float`
        The proportion of the dataset of each class to be in the train set.

    Returns:
    --------
    train_data : `list` of `Bio.SeqRecord.SeqRecord`
        Training dataset.
    train_labels : `list` 
        Corresponding training dataset labels.
    test_data : `list` of `Bio.SeqRecord.SeqRecord`
        Test dataset.
    test_labels : `list`
        Corresponding test dataset labels.      
    """   
    # Split data into classes
    class_data = collections.defaultdict(list)
    for i, label in enumerate(labels):
        class_data[label].append(data[i]) 
    
    split_data = collections.defaultdict(list)
    for i, (key, val) in enumerate(class_data.items()):
        num = len(val)
        train_lim = int(num*train_size)
        # Get data
        split_data["train_data"].extend(val[:train_lim])
        split_data["test_data"].extend(val[train_lim:])
        # Get corresponding labels
        split_data["train_labels"].extend([key]*train_lim)
        split_data["test_labels"].extend([key]*(num-train_lim))  
        
    # Shuffle data
    split_data["train_data"], split_data["train_labels"] = shuffle_data(split_data["train_data"], 
                                                                        split_data["train_labels"])
    split_data["test_data"], split_data["test_labels"] = shuffle_data(split_data["test_data"], 
                                                                      split_data["test_labels"])
    return split_data["train_data"], split_data["train_labels"], split_data["test_data"], split_data["test_labels"]