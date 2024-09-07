"""
This module is meant for extracting the data from the dataset, and it has some predefined format which includes
where to exclude the metadata, which columns to extract etc.
"""
import pandas as pd # pd imported and bound locally
import os
import utilities as ut

class DataExtractor:
    """
    Data extractor as the name suggests provide functionality to get data from different types of files for this project.
    It is mostly kept generic but in some cases it might be project specific.
    """
    def __init__(self, source):
        """
        Initialize the data extractor class by assigning the source folder path of the dataset.
        :param source: source folder path
        :type source: basestring
        """

        self.source = source

    def get_data_from_all_files_matching_id(self,search_keyword, columns_to_extract,ids, separator=';'):
        """
        It combines data which is extracted from all the csvs,
        those are present under the source folder which is set in the __init__ function.

        :return: data from all files combined in df for all given columns
        :rtype: df data frame

        **It is expected that all CSVs has exact same column names to be extracted, only NOT case-sensitive names.
        :param search_keyword:
        :type search_keyword:
        :param columns_to_extract:
        :type columns_to_extract:
        :param ids:
        :type ids:
        :param separator:
        :type separator:
        :return:
        :rtype:
        """
        if not os.path.isdir(self.source):
            raise ValueError(f"The source path {self.source} is not a valid directory.")

        id_list =  []
        csv_files = []
        df_list_return = []  # create an empty  list and push all the Xs to this list.


        for id in ids:
            # Get all id number's related csv files in the directory
            csv_file =  os.path.join(self.source, str(id) + '.csv')
            if not os.path.exists(csv_file):
                print(f"File not found: {csv_file}")
                continue  # Skip to the next file

            df_from_file = self.extract_columns_from_csv(csv_file, search_keyword, columns_to_extract, separator=';')
            if df_from_file is not None:
                df_from_file = df_from_file.drop(columns = [search_keyword])
                df_list_return.append(df_from_file)
                id_list.append(id)

        return df_list_return,id_list

    def get_data_from_all_files_under_source(self,search_keyword, columns_to_extract, separator=';'):

        """
        It combines data which is extracted from all the csvs,
        those are present under the source folder which is set in the __init__ function.

        :return: data from all files combined in df for all given columns
        :rtype: df data frame

        **It is expected that all CSVs has exact same column names to be extracted, only NOT case-sensitive names.
        """
        if not os.path.isdir(self.source):
            raise ValueError(f"The source path {self.source} is not a valid directory.")

        id_list =  []
        # Get all .csv files in the directory
        csv_files = [os.path.join(self.source, file) for file in os.listdir(self.source) if file.endswith('.csv')]

        df_list_return = [] # create an empty  list and push all the Xs to this list.
        for cf in csv_files:
            df_from_file = self.extract_columns_from_csv(cf, search_keyword, columns_to_extract, separator=';')
            if df_from_file is not None:
                df_from_file.drop(columns = [search_keyword])
                df_list_return.append(df_from_file)
                id_list.append(ut.get_file_name(cf))

        return df_list_return,id_list

    # Function to read the CSV, find the index line, and extract specific columns
    def extract_columns_from_csv(self,file_name, search_keyword, columns_to_extract, separator=';'):
        """
         Extracts given columns from the input file and also considers the input separator for the columns.
        :param file_name: Path of the file to be read.
        :type file_name: basestring
        :param search_keyword: search keyword from where the columns start.
        :type search_keyword: basestring
        :param columns_to_extract: list of the columns to be extracted from the file.
        :type columns_to_extract: list of basestring
        :param separator: separator for the csv file.
        :type separator: character
        :return: returns the data from the file.
        :rtype: pd dataframe
        """
        data_file_path = os.path.join(self.source, file_name)

        # Open the file and read it line by line
        with open(data_file_path, 'r') as file:
            lines = file.readlines()

        # Find the line containing the search keyword
        start_index = None
        for i, line in enumerate(lines):
            if search_keyword in line:
                start_index = i
                break

        # If the search keyword is not found, raise an exception
        if start_index is None:
            raise ValueError(f"The keyword '{search_keyword}' was not found in the file.")

        # Skip lines before the 'index' keyword and load the remaining data into a DataFrame
        df = pd.read_csv(data_file_path, skiprows = start_index, sep = separator, index_col = None)

        # Extract the specified columns
        extracted_df = df[columns_to_extract]

        return extracted_df
