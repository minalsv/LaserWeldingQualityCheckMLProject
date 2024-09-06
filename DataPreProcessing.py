import numpy as np
import pandas as pd

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

    def random_interpolation_within_range(self,column_to_be_interpolated, reference_columns):
        """
        Interpolate missing values in the 'column_to_be_interpolated' column by selecting
        a random value within the range of the nearest available data points.
        This is done within groups defined by reference_columns.
        """
        grouped = self.df.groupby(reference_columns) #grouping of values is done here so data gets interpolated within the group and then the range used for the interpolation is dependant on this group.


        def interpolate_with_random(group):
            """
            Interpolate the missing values with random numbers within the nearest neighbours value range.
            NOTE: The function has been nested here to limit its usage to the outer function.

            :param group:
            :type group:
            :return:
            :rtype:
            """
            non_nan_indices = group.dropna().index
            for idx in group.index:
                if pd.isna(group.loc[idx]):
                    prev_idx = non_nan_indices[non_nan_indices < idx].max() if len(
                        non_nan_indices[non_nan_indices < idx]) > 0 else None
                    next_idx = non_nan_indices[non_nan_indices > idx].min() if len(
                        non_nan_indices[non_nan_indices > idx]) > 0 else None

                    prev_value = group.loc[prev_idx] if prev_idx is not None else None
                    next_value = group.loc[next_idx] if next_idx is not None else None

                    if prev_value is not None and next_value is not None:
                        group.loc[idx] = np.random.uniform(prev_value, next_value)
                    elif prev_value is not None:
                        group.loc[idx] = prev_value
                    elif next_value is not None:
                        group.loc[idx] = next_value
            return group

        self.df[column_to_be_interpolated] = grouped[column_to_be_interpolated].apply(interpolate_with_random)


    def linearly_interpolate_column_values(self,column_to_be_interpolated, reference_columns):
        """

        :param column_to_be_interpolated:
        :type column_to_be_interpolated:
        :param reference_columns:
        :type reference_columns:
        :return:
        :rtype:
        """
        grouped = self.df.groupby(reference_columns)

        # Interpolate missing column_to_be_interpolated values within each group
        self.df[column_to_be_interpolated] = grouped[column_to_be_interpolated].transform(
            lambda group: group.interpolate(method = 'linear'))

        # Handle any remaining missing values with forward and backward fill
        self.df[column_to_be_interpolated].fillna(method = 'ffill', inplace = True)  # Forward fill as a fallback
        self.df[column_to_be_interpolated].fillna(method = 'bfill', inplace = True)  # Backward fill as a fallback

    def get_processed_data(self):
        """
        Returns the processed data.
        :return: processed data
        :rtype: pandas df
        """
        return self.df