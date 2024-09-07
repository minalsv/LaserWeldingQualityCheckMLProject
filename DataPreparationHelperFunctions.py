import pandas as pd
import os
import DataExtractor as de
import DataPlotter as dp
import utilities as ut
import config as cfg
from DataPreProcessing import DataPreProcessing

def get_test_and_train_meta_data():
    """

    :return:
    :rtype:
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
    ut.write_to_excel(interpolated_df, cfg.label_file_path, cfg.output_label_sheet)

    print("Get the test meta data")
    # Extract the data whch can be later used for the test purpose, remaining data can be used for training and validation
    test_info_df = interpolated_df[interpolated_df[cfg.column_name_which_decides_values_testing].notnull()]
    print(test_info_df.columns)
    print(test_info_df.info())

    print("Get the train and validation meta data")
    train_validation_df = interpolated_df[interpolated_df[cfg.column_name_which_decides_values_testing].isnull()]
    print(train_validation_df.columns)
    print(train_validation_df.info())

    print(
        f"Legnth of the test df {len(test_info_df)} + length of the train/validation info df {len(train_validation_df)} "
        f"is {len(test_info_df) + len(train_validation_df)} which is equal to the original info df i.e. {len(interpolated_df)}")

    modified_test_info_df, modified_train_validation_info_df = ut.transfer_excess_test_data(cfg.transfer_head_counts,
                                                                                            cfg.transfer_tail_counts,
                                                                                            test_info_df,
                                                                                            train_validation_df)
    print(
        f"Legnth of the modified test df {len(modified_test_info_df)} + length of the train/validation info df {len(modified_train_validation_info_df)} "
        f"is {len(modified_test_info_df) + len(modified_train_validation_info_df)} which is equal to the original info df i.e. {len(interpolated_df)}")

    return modified_test_info_df, modified_train_validation_info_df

def get_X_y_and_test_data_from_meta_data(modified_test_info_df, modified_train_validation_info_df):
    """

    :param modified_test_info_df:
    :type modified_test_info_df:
    :param modified_train_validation_info_df:
    :type modified_train_validation_info_df:
    :return:
    :rtype:
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

    # get IDs from train/validation info
    # get their "y" labels from info
    # categorise the labels into buckets and reassign against each ID (or in order)
    # get data from csv for each listed ID and create "y" in parallel
    # get test_X for each ID in the same sequence as in test_y


    return X,y,test_X,test_y