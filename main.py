"""
This app extracts the data from the specified dataset, plots and present the statistics along with the training a model, verifying etc. for the quality check of welding.
"""
import DataPreparationHelperFunctions as dphf
import utilities as ut

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # STEP 1 - get the data in the processable format
    # Extract meta information about the actual data
    test_info_df, train_validation_info_df = dphf.get_test_and_train_meta_data()
    # Extract data and labels for train and test sets.
    X,y,test_x,test_y = dphf.get_X_y_and_test_data_from_meta_data(test_info_df, train_validation_info_df) # do not change the input order
    print(X)

    # STEP 2 - Apply other pre-processing techniques e.g. normalise/standardise etc.

    # STEP 3  - Visualise

    # Plotting x data for two instances just for visual comparison, OK and NOT OK results
    ut.plot_both_classes_input_data(X,y)




    # STEP 4 - Do some adjustments if required based on the visualisation understanding

    # Step 5 - Run a simple Keras model

    # STEP 6 - Visualise the results

    # Step 7 - See the result and experiment with it.

    # Step 8 - compare STEP 5 and STEP 7 results.

    # STEP 9 - Run keras lib based model - functional model

    # STEP 10 - Visualise the results

    # STEP 10 - Adjust the parameters and visualise the results

    # STEP 11 - Compare results

    # STEP 12 - Run keras subclassed model

    # STEP 13 - Visualise the results

    # STEP 14 - Adjust the parameters and visualise the results

    # STEP 15 - Compare results

    # STEP 16 - Compare all models and conclude

    # End


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