import os

current_dir = os.path.dirname(__file__)
dataset_folder = os.path.join(current_dir, 'dataset/LWM/')
label_file_path = os.path.join(current_dir, 'dataset/Versuchsliste.xlsx')

index_column = 'Index'
search_keyword = index_column  # meta data ends here
columns_to_extract = ['P-Raw', 'T-Raw', 'R-Raw', 'L-Raw', index_column]  # Replace with your column names or indices

reference_columns = ['Power', 'Focus', 'Gap']
column_to_be_interpolated = 'Tension force'

input_label_sheet = 'input_labels_orig'
output_label_sheet = 'modified_labels'
