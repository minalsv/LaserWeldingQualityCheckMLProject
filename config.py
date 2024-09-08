import os

current_dir = os.path.dirname(__file__)
dataset_folder = os.path.join(current_dir, 'dataset/LWM/')
label_file_path = os.path.join(current_dir, 'dataset/Versuchsliste.xlsx')

index_column = 'Index'
search_keyword = index_column  # meta data ends here
columns_to_extract = ['P-Raw', 'T-Raw', 'R-Raw', 'L-Raw', index_column]  # Replace with your column names or indices
features = ['P-Raw', 'T-Raw', 'R-Raw', 'L-Raw']

#all columns from the modified sheet
all_columns = ['ID (Unnamed: 0_level_1)', 'Position außen [µOhm] (Unnamed: 1_level_1)',
       'Position mitte ( [µOhm])', 'Power ([W])', 'Focus ([mm])', 'Gap ([µm])',
       'Tension force ([N])', 'Versagensart (Unnamed: 7_level_1)',
       'Typ (Unnamed: 8_level_1)', 'Bemerkungen (Unnamed: 9_level_1)',
       'Bemerkungen (Unnamed: 10_level_1)']

reference_columns = ['Power ([W])','Focus ([mm])','Gap ([µm])']
column_to_be_interpolated = 'Tension force ([N])'
column_name_which_decides_values_testing = 'Versagensart (Unnamed: 7_level_1)'
input_label_sheet = 'input_labels_orig'
output_label_sheet = 'modified_labels'
id_column = 'ID (Unnamed: 0_level_1)'
header_rows = [0,1]

# we could move 20 from head and 20 from tail side of the test data to the train/validation data.
transfer_head_counts = 20
transfer_tail_counts = 20