import pandas as pd
import os
import DataExtractor as de
import utilities as ut
import config as cfg
from DataPreProcessing import DataPreProcessing
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

def get_test_and_train_meta_data():
    """
    Reads the top level dataset file which describes for which ID what is the label.
    Also, it gives info about which readings were actual by test and which are interpolated.
    Use interpolated readings for train/validation and actual readings for test purpose.
    But the number of test readings are going beyond 20% so moving few data points taken in actual tension test to the train/validation.
    This moving of data points is still happening at meta level and not the actual CSV data.
    Once the metadata is ready then we could use that to build the real test, train Xs and "y" labels.

    :return: Returns metadata for test and train
    :rtype: df,df
    """
    # first of all, prepare data by extrapolation and other techniques and then get Ys for all the IDs which are extracted above
    # we need to consider only those Ys for which we have data.

    y_df = ut.read_excel_to_df(cfg.label_file_path, cfg.input_label_sheet, cfg.header_rows)
    print(y_df.columns)
    print(y_df.head())
    column_adjusted_y_df = ut.adjust_columns(y_df)

    data_pre_processor = DataPreProcessing(column_adjusted_y_df)
    interpolated_df = data_pre_processor.interpolate_missing_column_values(cfg.column_to_be_interpolated,
                                                                           cfg.reference_columns)

    interpolated_df['Status'] = interpolated_df['Tension force ([N])'].apply(lambda x: 1 if x >= 3000 else 0)

    ut.write_to_excel(interpolated_df, cfg.label_file_path, cfg.output_label_sheet)

    print("Get the test meta data")
    # Extract the data which can be later used for the test purpose, remaining data can be used for training and validation
    test_info_df = interpolated_df[interpolated_df[cfg.column_name_which_decides_values_testing].notnull()]
    print(test_info_df.columns)
    print(test_info_df.info())

    print("Get the train and validation meta data")
    train_validation_df = interpolated_df[interpolated_df[cfg.column_name_which_decides_values_testing].isnull()]
    print(train_validation_df.columns)
    print(train_validation_df.info())

    print(
        f"Length of the test df {len(test_info_df)} + length of the train/validation info df {len(train_validation_df)} "
        f"is {len(test_info_df) + len(train_validation_df)} which is equal to the original info df i.e. {len(interpolated_df)}")

    modified_test_info_df, modified_train_validation_info_df = ut.transfer_excess_test_data(cfg.transfer_head_counts,
                                                                                            cfg.transfer_tail_counts,
                                                                                            test_info_df,
                                                                                            train_validation_df)
    print(
        f"Length of the modified test df {len(modified_test_info_df)} + length of the train/validation info df {len(modified_train_validation_info_df)} "
        f"is {len(modified_test_info_df) + len(modified_train_validation_info_df)} which is equal to the original info df i.e. {len(interpolated_df)}")

    return modified_test_info_df, modified_train_validation_info_df

def get_X_y_and_test_data_from_meta_data(test_meta_info_df, train_validation_meta_info_df):
    """
    Gets the meta info for test and train and then creates X data along with the labels "Y" from csv files and the master meta file.
    :param test_meta_info_df:  test meta information
    :type test_meta_info_df: df
    :param train_validation_meta_info_df: train/validation meta information
    :type train_validation_meta_info_df: df
    :return: Extracted X,y,test_X,test_y
    :rtype: df,df,df,df
    """
    X = None
    y = None
    test_X = None
    test_y = None

    # get IDs from test info
    # get their test_y label from info
    # categorise the labels into buckets and reassign against each ID (or in order)
    # get data from csv for each listed ID and create test_y in parallel
    # get test_X for each ID in the same sequence as in test_y
    data_extractor = de.DataExtractor(cfg.dataset_folder)
    test_X, test_ids_list = data_extractor.get_data_from_all_files_matching_id(cfg.search_keyword,
                                                                                             cfg.columns_to_extract,test_meta_info_df[cfg.id_column])
    test_y = ut.get_label_list_using_its_ids(test_meta_info_df,test_ids_list)
    #print(f"Extracted xs for test:\n ${test_X}")
    print(f"test Labels {len(test_y)}:\n ${test_y}")
    # get IDs from train/validation info
    # get their "y" labels from info
    # categorise the labels into buckets and reassign against each ID (or in order)
    # get data from csv for each listed ID and create "y" in parallel
    # get test_X for each ID in the same sequence as in test_y
    X, train_ids_list = data_extractor.get_data_from_all_files_matching_id(cfg.search_keyword,
                                                                                             cfg.columns_to_extract,train_validation_meta_info_df[cfg.id_column])
    y = ut.get_label_list_using_its_ids(train_validation_meta_info_df,train_ids_list)
    #print(f"Extracted xs for train/validation:\n ${test_X}")
    print(f"y Labels {len(y)}:\n ${y}")

    return X,y,test_X,test_y


def train_test_data(X,y,test_size= 0.2):
    """
    Divides the X and y into train and test using the provided split percentage and then normalises the data using the Standard scaler method.
    :param X: X data list
    :type X:  list of dfs
    :param y: labels
    :type y: list of booleans
    :param test_size: test data split e.g. 0.2 means 20% in the splitting.
    :type test_size: float
    :return: X_train,X_val, y_train, y_val
    :rtype: list of dfs, list of dfs, list of boolean, list of booleans
    """
    # Convert X (list of DataFrames) to a 3D NumPy array
    X_array = np.array([df.to_numpy() for df in X])  # Shape: (168, 20000, num_features)
    y_array = np.array(y)  # Shape: (168,)

    # Split the data into training and validation sets (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X_array, y_array, test_size=0.2, random_state=42)

    # Normalize the data (you can also apply normalization inside the KFold loop for each fold)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

    return X_train, X_val, y_train, y_val