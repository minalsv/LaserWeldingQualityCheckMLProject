"""
This app extracts the data from the specified dataset, plots and present the statistics along with the training a model, verifying etc. for the quality check of welding.
"""
import DataPreparationHelperFunctions as dphf

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    modified_test_info_df, modified_train_validation_info_df = dphf.get_test_and_train_meta_data()
    X,y,test_x,test_y = dphf.get_X_y_and_test_data_from_meta_data(modified_test_info_df, modified_train_validation_info_df)
    '''
    
        # Extract the columns from each file and then plot
    data_extractor = de.DataExtractor(cfg.dataset_folder)
    #extracted_data = data_extractor.extract_columns_from_csv(file_name, search_keyword, columns_to_extract)
    extracted_data_list,id_list = data_extractor.get_data_from_all_files_under_source( cfg.search_keyword, cfg.columns_to_extract)

    combined_df = pd.concat(extracted_data_list, ignore_index = True)
    print("Dataset Info \n",combined_df.head())
    print(f"{len(id_list)} IDs :\n${id_list}")

    X = combined_df.values # get X
    print(X)
    
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