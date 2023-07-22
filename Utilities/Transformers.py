import pandas as pd
import json
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder


# ToDo написать докстринги.

class MultilabelTransformer(BaseEstimator, TransformerMixin):
    '''
    Class for onehot encode data separated by comma.

    Attributes:
        dropna (bool): Flag for dropping column that represents None value.
    '''

    def __init__(self, dropna=False):
        """
        The constructor for DataPreprocessor class.

        Parameters:
           dropna (bool): Flag for dropping column that represents None value.
        """

        self.dropna = dropna

    def fit(self, X, y=None):
        """
        Fit MultilabelTransformer to X.

        Parameters:
           X (DataFrame): The data to determine the categories of each feature.
           y (None): Ignored. This parameter exists only for compatibility.
           
        Returns:
            Fitted transformer.
        """
        
        self.__vectorizer = CountVectorizer(analyzer=lambda x: set(x))
        self.__vectorizer.fit(
            X.apply(lambda x: str(x).replace(', ', ',').split(',')))
        if self.dropna:
            self.__feature_names = list(
                filter(lambda x: x != 'nan', self.__vectorizer.get_feature_names_out()))
        else:
            self.__feature_names = self.__vectorizer.get_feature_names_out()
        return self

    def transform(self, X, y=None):
        """
        Transform X using MultilabelTransformer.

        Parameters:
           X (DataFrame): The data to transform.
           
        Returns:
            Transformed input.
        """
        
        encoded_column = self.__vectorizer.transform(
            X.apply(lambda x: str(x).replace(', ', ',').split(','))).A
        column_names = list(map(
            lambda x: str(x), self.__vectorizer.get_feature_names_out()))
        onehot_df = pd.DataFrame(
            encoded_column, columns=column_names, index=X.index)

        if self.dropna:
            try:
                onehot_df.drop('nan',
                               axis=1, inplace=True)
            except KeyError:
                pass

        return onehot_df

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters:
           input_features (None): Ignored. This parameter exists only for compatibility.
           
        Returns:
            Transformed feature names.
        """
        
        return self.__feature_names


class LocationTransformer(BaseEstimator, TransformerMixin):
    '''
    Class for transform locations to postalcodes then onehot encode them.

    Attributes:
        locations_indexes (dict): Dictionary that defines postalcodes for locations.
        min_occurances (int): Number of postalcode occurances, below which it wont be accounted.
    '''

    def __init__(self, locations_postalcodes, min_occurances=5):
        """
        The constructor for LocationTransformer class.

        Parameters:
            locations_indexes (dict): Dictionary that defines postalcodes for locations.
            min_occurances (int): Number of postalcode occurances, below which it wont be accounted.
        """

        self.min_occurances = min_occurances
        self.locations_postalcodes = locations_postalcodes

    def fit(self, X, y=None):
        """
        Fit LocationTransformer to X.

        Parameters:
           X (DataFrame): The data to determine the categories of each feature.
           y (None): Ignored. This parameter exists only for compatibility.
           
        Returns:
            Fitted transformer.
        """
        
        self.__postalcode_encoder = OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore') \
            .fit(self.__preprocess(X))
        return self

    def transform(self, X, y=None):
        """
        Transform X using postalcode encoder.

        Parameters:
           X (DataFrame): The data to transform.
           
        Returns:
            Transformed input.
        """
        
        temp_df = self.__postalcode_encoder.transform(self.__map_postalcodes(X))
        return temp_df

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters:
           input_features (None): Ignored. This parameter exists only for compatibility.
           
        Returns:
            Transformed feature names.
        """
        
        return self.__postalcode_encoder.get_feature_names_out()

    def __preprocess(self, X):
        """
        Preprocess X before fitting.

        Parameters:
           X (DataFrame): The data to preprocessed.
           
        Returns:
            Preprocessed input.
        """
        
        temp_df = self.__map_postalcodes(X)
        indexes = temp_df.groupby('Почтовый индекс')['Почтовый индекс'] \
            .agg('count').sort_values(ascending=False)
        temp_df['Почтовый индекс'] = temp_df['Почтовый индекс'].apply(
            lambda x: x if x in indexes[indexes >= self.min_occurances] else 'другой')
        return temp_df
    
    def __map_postalcodes(self, X):
        """
        Mapping X into postalcodes.

        Parameters:
           X (DataFrame): Locations to be mapped.
           
        Returns:
            Postalcodes.
        """
        
        temp_df = pd.DataFrame()
        temp_df['Почтовый индекс'] = X.apply(
            lambda x: self.locations_postalcodes.get(x))
        return temp_df


class ParkingTransformer(BaseEstimator, TransformerMixin):
    '''
    Class for transform data about parking.
    '''

    def fit(self, X, y=None):
        """
        Fit ParkingTransformer to X.

        Parameters:
           X (DataFrame): The data to determine the categories of each feature.
           y (None): Ignored. This parameter exists only for compatibility.
           
        Returns:
            Fitted transformer.
        """
        
        self.__parking_encoder = OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore') \
            .fit(self.__preprocess(X))
        return self

    def transform(self, X, y=None):
        """
        Transform X using parking encoder.

        Parameters:
           X (DataFrame): The data to transform.
           
        Returns:
            Transformed input.
        """
        
        temp_df = self.__parking_encoder.transform(self.__preprocess(X))
        self.__feature_names = self.__parking_encoder.get_feature_names_out()
        return temp_df

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters:
           input_features (None): Ignored. This parameter exists only for compatibility.
           
        Returns:
            Transformed feature names.
        """
        
        return self.__feature_names

    def __preprocess(self, X):
        """
        Preprocess X to match the following transformations.

        Parameters:
           X (DataFrame): The data to preprocessed.
           
        Returns:
            Preprocessed input.
        """
        
        temp_df = pd.DataFrame()
        temp_df['Парковка'] = X
        temp_df['Парковка'] = temp_df['Парковка'].apply(lambda x: 'не указана' if str(x) == 'nan' else (
            x if x == 'за шлагбаумом во дворе' else 'другая'))
        return temp_df


class ScalerTransformer(BaseEstimator, TransformerMixin):
    '''
    Class for transform locations to postalcodes then onehot encode them.

    Attributes:
        transformer (TransformerMixin): Transformer to perform scaling operations.
    '''

    def __init__(self, transformer):
        """
        The constructor for ScalerTransformer class.

        Parameters:
            transformer (TransformerMixin): Transformer to perform scaling operations.
        """

        self.transformer = transformer

    def fit(self, X, y=None):
        """
        Fit ScalerTransformer to X.

        Parameters:
           X (DataFrame): The data to determine the categories of each feature.
           y (None): Ignored. This parameter exists only for compatibility.
           
        Returns:
            Fitted transformer.
        """
        
        self.transformer.fit(X)
        return self

    def transform(self, X, y=None):
        """
        Transform X using scaling transformer.

        Parameters:
           X (DataFrame): The data to transform.
           
        Returns:
            Transformed input.
        """
        
        self.__feature_names = X.columns
        return pd.DataFrame(self.transformer.transform(X), columns=X.columns)

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters:
           input_features (None): Ignored. This parameter exists only for compatibility.
           
        Returns:
            Transformed feature names.
        """
        
        return self.__feature_names


class FeatureSelector(BaseEstimator, TransformerMixin):
    '''
    Class for selecting features.

    Attributes:
        columns (list): List of columns to select featuers.
    '''

    def __init__(self, columns):
        """
        The constructor for FeatureSelector class.

        Parameters:
            columns (list): List of columns to select featuers.
        """

        self.columns = columns

    def fit(self, X, y=None):
        """
        This method exists only for compatibility.

        Parameters:
           X (DataFrame): This parameter exists only for compatibility.
           y (None): Ignored. This parameter exists only for compatibility.
           
        Returns:
            Self.
        """
        
        return self

    def transform(self, X, y=None):
        """
        Transform X using list of selected columns.

        Parameters:
           X (DataFrame): The data to transform.
           
        Returns:
            Transformed input.
        """
        
        return X[self.columns]

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters:
           input_features (None): Ignored. This parameter exists only for compatibility.
           
        Returns:
            Transformed feature names.
        """
        
        return self.columns
