
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from Utilities.Transformers import ScalerTransformer
from Utilities.Transformers import FeatureSelector


class DataPreprocessor():
    '''
    Class for preprocessing data before using ml model.

    Attributes:
        preprocessor (Pipeline): Pipeline for data preprocessing.
    '''
    
    # def mainify(self, obj):
    #     """If obj is not defined in __main__ then redefine it in 
    #     main so that dill will serialize the definition along with the object"""
    #     if obj.__module__ != "__main__":
    #         import __main__
    #         import inspect
    #         s = inspect.getsource(obj)
    #         co = compile(s, '<string>', 'exec')
    #         exec(co, __main__.__dict__)
        
    def __init__(self, column_preprocessor, selected_columns):
        """
        The constructor for DataPreprocessor class.

        Parameters:
           column_preprocessor (ColumnTransformer): Column transformer for preprocessing data before using feature selector.
           selected_columns (list): List of column names to select.   
        """
        
        self.__column_preprocessor = column_preprocessor
        self.__selected_columns = selected_columns

        # self.mainify(FeatureSelector)
        # self.mainify(ScalerTransformer)

        self.preprocessor = Pipeline(
            steps=[
                ('column_preprocessor', self.__column_preprocessor),
                ('feature_selector', FeatureSelector(self.__selected_columns)),
                ('feature_scaler', ScalerTransformer(MinMaxScaler()))
            ]
        )

    def fit(self, df):
        """
        Fitting data preprocessing pipeline on given data.

        Parameters:
           df (DataFrame): Data to fit data preprocessing pipeline.
        """

        self.preprocessor.fit(df)
        return self

    def transform(self, df):
        """
        Transforming data to a format that ml model uses.

        Parameters:
            df (DataFrame): Data to be transformed.

        Returns:
            Data in format that ml model uses.
        """
        
        return self.preprocessor.transform(df)

    def inverse_predicted_values(self, values):
        """
        Invert 1D list of values by the feature scaler.

        Parameters:
            values (list): 1D list of values to be inverted.

        Returns:
            1D list of inverted by feature scaler values.
        """
        
        transformer = self.preprocessor['feature_scaler'].transformer
        temp_df = pd.DataFrame(
            np.zeros((len(values), transformer.n_features_in_)))
        temp_df[0] = values
        
        unscaled = transformer.inverse_transform(temp_df)[:, :1].flatten()
        return self.preprocessor['column_preprocessor'].named_transformers_['price'].inverse_transform(unscaled)
