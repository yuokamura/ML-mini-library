# plan:
# read pickel file
# pre processing
# tuning on the prameter 
# train the model
# save the model 
# validation
# prediction on the new data


import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder

class FileToDataframe:
    def __init__(self):
        ...

    def create_dataframe(self, path:str) -> pd.DataFrame:
        dataframe = pd.read_pickle(path)
        return dataframe

class PreProcessingCategoricalEncoder:
    def __init__(self):
        self.cat_weights=None
        self.encoder=None

    def fit_category(self,dataframe) -> pd.DataFrame:
        # Select categorical columns.  though maybe better to rely input from the user?
        categorical_columns = dataframe.select_dtypes(include='object').columns.tolist() 

        # Create and fit the OneHotEncoder
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.encoder.fit(dataframe[categorical_columns])

        # Save the categorical weights
        self.cat_weights = self.encoder.categories_
        
        return self.cat_weights

    def transform_category(self, dataframe) -> pd.DataFrame:
        # Select categorical columns
        categorical_columns = dataframe.select_dtypes(include='object').columns.tolist()

        # Transform the categorical columns using the saved encoder
        encoded_data = self.encoder.transform(dataframe[categorical_columns])

        # Create a new DataFrame with the encoded data
        encoded_dataframe = pd.DataFrame(encoded_data, columns=self.encoder.get_feature_names(categorical_columns))

        # Concatenate the encoded DataFrame with the remaining columns
        encoded_dataframe = pd.concat([dataframe.drop(columns=categorical_columns), encoded_dataframe], axis=1)

        return encoded_dataframe