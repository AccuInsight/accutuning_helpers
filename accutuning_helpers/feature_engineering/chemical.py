import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, QED, MolFromSmiles, MolFromInchi, MolToSmiles, Descriptors, MolSurf, rdMolDescriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.Crippen import MolLogP 
from rdkit.Chem.QED import qed
from sklearn.base import BaseEstimator, TransformerMixin
import transformers
import logging

logger = logging.getLogger()

class AccutuningChemicalConverter(BaseEstimator, TransformerMixin):
    """
    Transform Chemical molecule string into vectors or tokens using Rdkit or transformers pretrained model.
    
    Parameters
    ----------
    feature_names : list
        List of feature names (str) to convert
    feature_format : str, default='smiles'
        Molecule formats, 'smiles' or 'inchi'
    how : str, default='ecfp'
        Method to convert the molecule string, 'ecfp', 'fcfp' or 'tokenize'
    dim : int, default=512
        Dimension for the molecule conversion
        Used only if 'how' is 'ecfp' or 'fcfp'
    add_properties : list, default=[]
        Add chemical properties of the input molecule columns for each.
        'QED', 'MW', 'ALOGP', 'HBA', 'HBD', 'PSA' ...
    """

    def __init__(self, feature_names, feature_format='smiles', how='ecfp', dim=512, add_properties=None):
        self.feature_names = feature_names
        self.feature_format = feature_format
        self.how = how
        self.dim = dim
        self.add_properties = add_properties

    def fit(self, X, y=0, **fit_params):
        
        if self.how == 'tokenize': 
            from transformers import AutoTokenizer
            self.converter = AutoTokenizer.from_pretrained('seyonec/ChemBERTa_zinc250k_v2_40k', use_fast=True)
            self.pad = self.converter.pad_token_id
        else:
            self.converter = self.how
        
        return self

    def transform(self, X, y=0):
        X_tr = X.copy()
        for feature in self.feature_names:
            target_column = X.loc[:, feature]
            target_idx = X.columns.get_loc(feature)
            X_front = X.columns[:target_idx]
            X_back = X.columns[target_idx + 1:]
        
            converted = target_column.map(lambda x: self._convert_mol(x, 
                                                                self.feature_format, 
                                                                self.converter,
                                                                self.dim))
            
            f = lambda x: '{}_{}'.format(feature, x)            
            converted = pd.DataFrame(converted.tolist()).rename(columns=f)
            if converted.isnull().values.any():
                converted.fillna(self.pad, inplace=True, downcast='infer')

            X_tr = pd.concat([X_tr[X_front], converted, X_tr[X_back]], axis=1)
            
            if self.add_properties:
                props = target_column.map(lambda x: self._add_properties(x, 
                                                                    self.feature_format,
                                                                    self.add_properties))
                X_tr = X_tr.join(pd.DataFrame.from_records(props, columns=self.add_properties))

        return X_tr

    def _convert_mol(self, sequence, feature_format, how, dim):
        if feature_format == 'smiles':
            m = MolFromSmiles(sequence)
        elif feature_format == 'inchi':
            m = MolFromInchi(sequence)
        else:
            logger.critical('AutoinsightChemicalConverter: Not a valid Molecule format selected.')
            m = None
        
        if how == 'ecfp':
            try:
                vec = GetMorganFingerprintAsBitVect(m, 2, nBits = dim)
                vec = vec.ToBitString()
            except Exception as e:
                vec = [0] * self.dim
        elif how == 'fcfp':
            try:
                vec = GetMorganFingerprintAsBitVect(m, 2, useFeatures = True, nBits = dim)
                vec = vec.ToBitString()
            except Exception as e:
                vec = [0] * self.dim
        elif isinstance(how, transformers.RobertaTokenizerFast):
            smiles = MolToSmiles(m)
            vec = how(smiles)['input_ids']
        else:
            logger.critical('AutoinsightChemicalConverter: No proper conversion method provided')
        
        converted = list(vec)
        
        return converted
    
    def _add_properties(self, sequence, feature_format, properties):
        props = []
        
        if feature_format == 'smiles':
            m = MolFromSmiles(sequence)
        elif feature_format == 'inchi':
            m = MolFromInchi(sequence)
        else:
            logger.critical('AutoinsightChemicalConverter: Not a valid Molecule format selected.')
            m = None
        
        try:
            pp = QED.properties(m)
            if properties:
                for i in properties:
                    if i == 'MW': 
                        props.append(round(pp.MW, 2))
                    elif i == 'ALOGP':
                        props.append(round(pp.ALOGP, 2))
                    elif i == 'HBA':
                        props.append(round(pp.HBA, 2))
                    elif i == 'HBD':
                        props.append(round(pp.HBD, 2))
                    elif i == 'PSA':
                        props.append(round(pp.PSA, 2))
                    elif i == 'QED':
                        props.append(round(QED.qed(m), 2))
                    else:
                        logger.critical(f'AutoinsightChemicalConverter: Not a property available: {i}')
        except Exception as e:
            props = [0] * len(properties)
            logger.critical(f'AutoinsightChemicalConverter: Fail to get properties from smiles: {mol}')
            logger.critical(e, exc_info=True)

        return props
