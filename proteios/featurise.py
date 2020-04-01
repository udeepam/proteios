import numpy as np
import pandas as pd
import collections

from Bio.SeqUtils.ProtParam import ProteinAnalysis

class Featuriser:
    def __init__(self, data, dicts):
        """
        Arguments:
        ----------
        data : `list` of `Bio.SeqRecord.SeqRecord`
            Data to fit the class with to get normalisations.
        dicts : `dict`
            Dictionaries.
        """
        self.labels = ["length", "molecular_weight", "isoelectric_point", "aromaticity", 
                       "instability_index", "gravy", "helix", "turn", "sheet", "reduced", "oxidised", 
                       "hydrophobicity", "flexibility", "hydrophilicity", "surface_accessibility", 
                       "janin", "charge_at_ph1", "charge_at_ph7", "charge_at_ph12"]        
        self.dicts = dicts
        # Fit class with data
        self.fit(data)

    def fit(self, data):
        """
        Parameters:
        -----------
        data : `list` of `Bio.SeqRecord.SeqRecord`
            Data to fit the class with to get normalisations.
        """
        # Featurise data
        featurised_data = self.featurise(data)
        # Get normalisations from data
        self.means = dict()
        self.stds = dict()
        for i in self.labels:
            self.means[i] = featurised_data[i].mean()
            self.stds[i] = featurised_data[i].std()
        
    def transform(self, data): 
        """
        Featurises the data and then normalises.
        
        Parameters:
        -----------
        data : `list` of `Bio.SeqRecord.SeqRecord`
            Data to fit the class with to get normalisations.
            
        Returns:
        --------
        featurised_data : `pandas.DataFrame`
            (num_data, features) The featurised and normalised data.
        """
        # Featurise data
        featurised_data = self.featurise(data)
        # Normalise data
        featurised_data = self.normalise(featurised_data)
        return featurised_data
        
    def normalise(self, data):
        """
        Normalises the data by subtracting the mean and dividing 
        by the standard deviation.
        
        Parameters:
        -----------
        data : `pandas.DataFrame`
            (num_data, features) The data to be normalised.   
            
        Returns:
        --------
        normalised_data : `pandas.DataFrame`
            (num_data, features) The normalised data.            
        """        
        # Normalise features by subtract mean and divide by standard deviation
        for i in self.labels:
            data[i] = (data[i]-self.means[i])/self.stds[i]    
        return data
    
    def featurise(self, data):
        """
        Featurise the data.

        Parameters:
        -----------
        data : `list` of `Bio.SeqRecord.SeqRecord`
            The data to be featurised.

        Returns:
        -------
        featurised_data : `pandas.DataFrame`
            (num_data, features) The featurised data.
        """
        # Get features of data
        features = collections.defaultdict(list)
        
        # Featurise the data
        for i, example in enumerate(data):
            # Convert Bio.SeqRecord.SeqRecord object to string for Bio.SeqUtils.ProtParam.ProteinAnalysis
            analysed_example = ProteinAnalysis(str(example.seq))

            features["length"].append(analysed_example.length)
            features["molecular_weight"].append(analysed_example.molecular_weight()) 
            features["isoelectric_point"].append(analysed_example.isoelectric_point()) 
            features["aromaticity"].append(analysed_example.aromaticity()) 
            features["instability_index"].append(analysed_example.instability_index()) 
            features["gravy"].append(analysed_example.gravy()) 

            reduced, oxidised = analysed_example.molar_extinction_coefficient()
            features["reduced"].append(reduced)
            features["oxidised"].append(oxidised)        

            helix, turn, sheet = analysed_example.secondary_structure_fraction()
            features["helix"].append(helix)
            features["turn"].append(turn)
            features["sheet"].append(sheet)

            features["charge_at_ph1"].append(analysed_example.charge_at_pH(1)) 
            features["charge_at_ph7"].append(analysed_example.charge_at_pH(7)) 
            features["charge_at_ph12"].append(analysed_example.charge_at_pH(12))                  

            features["hydrophobicity"].append(np.mean(analysed_example.protein_scale(self.dicts['kd'], window=5, edge=1.0)))
            features["flexibility"].append(np.mean(analysed_example.protein_scale(self.dicts['flex'], window=5, edge=1.0)))
            features["hydrophilicity"].append(np.mean(analysed_example.protein_scale(self.dicts['hw'], window=5, edge=1.0)))
            features["surface_accessibility"].append(np.mean(analysed_example.protein_scale(self.dicts['em'], window=5, edge=1.0)))
            features["janin"].append(np.mean(analysed_example.protein_scale(self.dicts['ja'], window=5, edge=1.0)))
    #         features["dipeptide_dg "].append(np.mean(analysed_example.protein_scale(self.dicts['diwv'], window=5, edge=1.0)))

            for key, val in analysed_example.get_amino_acids_percent().items():
                features[key].append(val*5)       
            first_50_analysed_example = ProteinAnalysis(str(example.seq)[:50])
            for key, val in first_50_analysed_example.get_amino_acids_percent().items():
                features["first_50_"+str(key)].append(val*5)
            last_50_analysed_example = ProteinAnalysis(str(example.seq)[-50:])
            for key, val in last_50_analysed_example.get_amino_acids_percent().items():
                features["last_50_"+str(key)].append(val*5)            

        return pd.DataFrame.from_dict(features)