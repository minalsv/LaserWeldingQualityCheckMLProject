import os
import pandas as pd
'''
Utilities has some generic helper functions which can be used in all the modules under this project.
'''
def get_file_name(file_name):
    """
    Remove the path and the extension of the file.
    :param file_name: name of the file with or without path.
    :type file_name: basestring
    :return: name of the file without the extension and path.
    :rtype: basestring
    """
    if os.path.dirname(file_name):
        # Extract file name without extension and remove path
        return os.path.splitext(os.path.basename(file_name))[0]
    else:
        # Only remove the extension
        return os.path.splitext(file_name)[0]


def filter_list_item(item_list, item_to_filter):
    """
    Filters the item from the given list and returns the filtered list.
    :param item_list: list of items to be filtered
    :type item_list: list
    :param item_to_filter: item to be filtered from the list.
    :type item_to_filter: list parameter
    """
    # Exclude the name using list comprehension
    filtered_items = [item for item in item_list if item != item_to_filter]

    return filtered_items

def write_to_excel(self, excel_file_path, new_sheet_name):
    """

    :param self:
    :type self:
    :param excel_file_path:
    :type excel_file_path:
    :param new_sheet_name:
    :type new_sheet_name:
    :return:
    :rtype:
    """
    with pd.ExcelWriter(excel_file_path, engine = 'openpyxl', mode = 'a') as writer:
        self.df.to_excel(writer, sheet_name = new_sheet_name, index = False)


def read_excel_to_df(excel_file_path, sheet_name):
    """
    Reads data from Excel into pandas dataframe format.
    :param excel_file_path: path+file name
    :type excel_file_path: basestring
    :param sheet_name: the sheet from which the data should be returned.
    :type sheet_name: basestring
    :return: all data from the input file and the sheet
    :rtype: pandas df
    """
    try:
        df = pd.read_excel(excel_file_path, sheet_name = sheet_name)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{excel_file_path}' was not found.")
    except ValueError:
        print(f"Error: The sheet '{sheet_name}' does not exist in the file '{excel_file_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")