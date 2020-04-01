import numpy as np

def shuffle_data(x, y):
    """
    Shuffles two lists of same length in the same way.
    
    Parameters:
    -----------
    x : `list`
        First list to be shuffled.
    y : `list`
        Second list to be shuffled.
        
    Returns:
    -------
    x : `list`
        Shuffled first list.
    y : `list`
        Shuffled second list.
    """
    data = list(zip(x, y))
    np.random.shuffle(data)
    x, y = zip(*data)
    return np.array(x), np.array(y)