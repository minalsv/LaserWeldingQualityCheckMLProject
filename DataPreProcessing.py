import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

class DataPreProcessing:
    """
    A class useful for pre-processing of the data. It is written in a generic way but some part could be project specific.
    """
    def __init__(self, dataframe):
        """
        Set the data which is to be processed in the initialisation as all the functions are called upon this set data.
        :param dataframe: data to be processed
        :type dataframe: pandas df
        """
        self.df = dataframe


    def get_processed_data(self):
        """
        Returns the processed data.
        :return: processed data
        :rtype: pandas df
        """
        return self.df


    def interpolate_missing_column_values(self,column_to_be_interpolated, reference_columns,interpolation_method='random'):
        """

        :param column_to_be_interpolated:
        :type column_to_be_interpolated:
        :param reference_columns:
        :type reference_columns:
        :return:
        :rtype:
        """
        # Backup the original column order
        original_columns = self.df.columns
        print(self.df.columns)

        # Group by function used to group(but also splits it into smaller df) the values based on the reference columns values
        grouped = self.df.groupby(reference_columns)

        # Apply random interpolation to each group, passing the column name as an argument
        self.df = grouped.apply(
            lambda group: self.interpolate_within_group(group, column_to_be_interpolated,interpolation_method)).reset_index(drop = True)

        # Restore the original column order
        self.df = self.df[original_columns]

        return self.df

    def interpolate_within_group(self, group, column_to_be_interpolated,interpolation_method='random'):
        """

        :param group: group created based on common values of the reference columns
        :type group: DataFrameGroupBy object
        :param column_to_be_interpolated:
        :type column_to_be_interpolated:
        :param interpolation_method:
        :type interpolation_method:
        :return:
        :rtype:
        """
        # Get all non-missing from the column_to_be_interpolated
        non_missing = group[column_to_be_interpolated].dropna()

        # If there are no non-missing values, return the group as is
        if non_missing.empty:
            return group
        if interpolation_method == 'random':
            # Perform random interpolation within the range of non-missing values
            min_value = non_missing.min()
            max_value = non_missing.max()

            group[column_to_be_interpolated] = group[column_to_be_interpolated].apply(
                lambda x: np.random.uniform(min_value, max_value) if pd.isna(x) else x
            )
        elif interpolation_method == 'linear':
            # Perform linear interpolation
            group[column_to_be_interpolated] = group[column_to_be_interpolated].interpolate(method='linear')


        return group

    def normalize_list_of_dfs(self,X):
        # Apply normalization to each DataFrame in the list
        normalized_X = [self.apply_min_max_scaler(df) for df in X]
    
        return normalized_X

    def apply_min_max_scaler(self,df):
        """
        Apply min-max scaler to the dataframe.
        :return: Normalised data
        :rtype: df
        """
        # Min-Max Normalization
        min_max_scaler = MinMaxScaler()
        data_normalized = pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)
        return data_normalized

    def apply_z_score(self):
        """
        Applies Z-score normalisation on the data.
        :return: Z scored normalised data
        :rtype: df
        """
        # Z-Score Normalization (Standardization)
        standard_scaler = StandardScaler()
        data_standardized = pd.DataFrame(standard_scaler.fit_transform(self.df), columns=self.df.columns)
        return data_standardized

    def apply_robust_scaling(self):
        """
        Apply Robust scaling on the data.
        :return: Scaled data.
        :rtype: df
        """
        # Robust Scaling
        robust_scaler = RobustScaler()
        data_robust_scaled = pd.DataFrame(robust_scaler.fit_transform(self.df), columns=self.df.columns)
        return data_robust_scaled