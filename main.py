"""
This app extracts the data from the specified dataset, plots and present the statistics along with the training a model, verifying etc. for the quality check of welding.
"""
import pandas as pd
import os
import DataExtractor as de
import DataPlotter as dp
import utilities as ut
import config as cfg

# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    # Extract the columns from each file and then plot
    data_extractor = de.DataExtractor(cfg.dataset_folder)
    #extracted_data = data_extractor.extract_columns_from_csv(file_name, search_keyword, columns_to_extract)
    extracted_data_list,id_list = data_extractor.get_data_from_all_files_under_source( cfg.search_keyword, cfg.columns_to_extract)

    combined_df = pd.concat(extracted_data_list, ignore_index = True)
    print("Dataset Info \n",combined_df.head())
    print(f"{len(id_list)} IDs :\n${id_list}")

    X = combined_df.values # get X
    print(X)
    # first of all, prepare data by extrapolation and other techniques and then get Ys for all the IDs which are extracted above
    # we need to consider only those Ys for which we have data.

    y_df = ut.read_excel_to_df(cfg.label_file_path, cfg.input_label_sheet)
    print(y_df)
    '''
    #y = extract_labels()
    # Create an instance of DataPlotter with the extracted data and a selected color palette
    plotter = dp.DataPlotter(extracted_data, palette_name = 'coolwarm')

    # Plot the data with different types using the selected color palette
    plotter.plot_data(graph_type = 'line', y_col = columns_to_extract[0], x_col = columns_to_extract[4], linestyle = '--', marker = 'x')
    #plotter.plot_data(graph_type = 'scatter', y_col = columns_to_extract[0], x_col = columns_to_extract[4])
    #plotter.plot_data(graph_type = 'bar', y_col = columns_to_extract[0], x_col = columns_to_extract[4])
    x_cols = ut.filter_list_item(columns_to_extract, index_column)
    plotter.plot_data(graph_type = 'histogram', x_col = x_cols, bins = 15)
    #plotter.plot_data(graph_type = 'boxplot', x_col = columns_to_extract[0])

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
    '''