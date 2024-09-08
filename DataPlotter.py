"""
The data plotter allows to plot the data by setting various color palettes and type of graphs.
"""
import matplotlib
#matplotlib.use('TkAgg')  # Choose an interactive backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

'''
color palettes 
Categorical Palettes: For categorical data (discrete groups or categories) - deep, pastel, dark, colorblind
Sequential Palettes: For data that has an inherent order, from low to high values -  Blues, BuGn, Greens, Oranges, Purples, Reds, YlGnBu, YlOrRd 
Diverging Palettes: For data that deviates around a central value (e.g., temperature deviations) - coolwarm, RdBu, BrBG, PiYG, PuOr, Spectral
'''
class DataPlotter:
    """
    It's a class for plotting the data by allowing to set palette, data values and in future the type too.
    """

    def __init__(self, data, palette_name='deep'):
        self.data = data
        self.palette = sns.color_palette(palette_name)

    def plot_line(self, x_col, y_col, linestyle='-', marker='o'):
        """Plot data as a line graph using the selected color palette."""
        plt.plot(self.data[x_col], self.data[y_col], marker=marker, color=self.palette[0], linestyle=linestyle)
        plt.title('Line Plot')
        mean_value = self.data[y_col].mean()
        median_value = self.data[y_col].median()
        plt.axhline(y=mean_value, color='r', linestyle='--', label=f'Mean {y_col}')
        plt.axhline(y=median_value, color='g', linestyle='-', label=f'Median {y_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)

    def plot_scatter(self, x_col, y_col):
        """Plot data as a scatter plot using the selected color palette."""
        plt.scatter(self.data[x_col], self.data[y_col], color=self.palette[1])
        plt.title('Scatter Plot')
        plt.xlabel(x_col)
        plt.ylabel(y_col)

    def plot_bar(self, x_col, y_col):
        """Plot data as a bar graph using the selected color palette."""
        plt.bar(self.data[x_col], self.data[y_col], color=self.palette[2])
        plt.title('Bar Plot')
        mean_value = self.data[y_col].mean()
        median_value = self.data[y_col].median()
        plt.axhline(y=mean_value, color='r', linestyle='--', label=f'Mean {y_col}')
        plt.axhline(y=median_value, color='g', linestyle='-', label=f'Median {y_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)

    def plot_histogram(self, x_col, bins=10):
        """
        Plots histogram of all present columns in the set data.
        :param x_col: columns to be plotted
        :type x_col: list of column names
        :param bins: bin size, set to 10 as default
        :type bins: int
        """
        plt.figure(figsize=(20, 15))
        for i,c in enumerate(x_col,1):
            plt.subplot(4, 4, i)
            sns.histplot(self.data[x_col], kde=True)
            plt.title(c)
        plt.tight_layout()
        plt.show()
        '''
        """Plot data as a histogram using the selected color palette."""
        plt.hist(self.data[col], bins=bins, color=self.palette[3])
        plt.title('Histogram')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        '''
    def plot_boxplot(self, col):
        """Plot data as a boxplot using the selected color palette."""
        sns.boxplot(data=self.data, x=col, color=self.palette[4])
        plt.title('Boxplot')
        plt.xlabel(col)

    def plot_data(self, graph_type, x_col=None, y_col=None, **kwargs):
        """Choose the plot type and call the appropriate method with customization options."""
        if graph_type == 'line':
            self.plot_line(x_col, y_col, **kwargs)
        elif graph_type == 'scatter':
            self.plot_scatter(x_col, y_col)
        elif graph_type == 'bar':
            self.plot_bar(x_col, y_col)
        elif graph_type == 'histogram':
            self.plot_histogram(x_col, **kwargs)
        elif graph_type == 'boxplot':
            self.plot_boxplot(x_col)
        else:
            raise ValueError(f"Unsupported graph type: {graph_type}")

        # Show the plot
        plt.show()

    def plot_histogram_for_all_input_features(self, df, features_to_be_plotted, graph_title, bins=15):
        """

        :param bins:
        :type bins:
        :param features_to_be_plotted:
        :type features_to_be_plotted:
        :param df:
        :type df:
        :param graph_title:
        :type graph_title:
        :return:
        :rtype:
        """
        # Plot histograms for each feature in the combined dataset
        fig, axs = plt.subplots(1, len(features_to_be_plotted), figsize = (15, 5))


        for i in range(len(features_to_be_plotted)):
            axs[i].hist(df[:2000, features_to_be_plotted[i]], bins = 50, alpha = 0.7, color = 'blue')
            axs[i].set_title(f'Histogram of {features_to_be_plotted[i]}')
            axs[i].set_xlabel('Value')
            axs[i].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    def plot_feature_with_stats(self,x_df,df_name):
        """
        Plots the linear graphs for all the columns of the input data with their mean and median but in a separate graph.
        :param x_df: input data
        :type x_df: df
        :param df_name: Name for the graph
        :type df_name:basestring
        """
        for i, column in enumerate(x_df.columns):
            plt.figure(figsize = (10, 5))

            # Plot the feature with a unique color
            plt.plot(x_df[column], label = column, color = self.palette[i % len(self.palette)])

            # Calculate mean and median
            mean_val = x_df[column].mean()
            median_val = x_df[column].median()

            median_color = 'green'
            mean_color = 'red'
            # Plot mean and median as horizontal lines
            plt.axhline(mean_val, color = mean_color, linestyle = '--', label = f'Mean: {mean_val:.2f}')
            plt.axhline(median_val, color = median_color, linestyle = '--', label = f'Median: {median_val:.2f}')

            # Annotate mean and median
            plt.text(len(x_df) - 1, mean_val, f'Mean: {mean_val:.2f}', color = mean_color, verticalalignment = 'center')
            plt.text(len(x_df) - 1, median_val, f'Median: {median_val:.2f}', color = median_color,
                     verticalalignment = 'center')

            # Title and labels
            plt.title(f'Feature: {column} for {df_name}')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            plt.show()


    def plot_all_features_with_stats_in_one_graph(self,df,df_name):
        """
        Plot data(all features) from the input dataframe in one graph.
        :param df: data to be plotted
        :type df: df
        :param df_name: Name to be given for the graph
        :type df_name: basestring
        """

        # Number of columns (features) in the DataFrame
        num_columns = len(df.columns)

        # Create subplots, one for each feature
        fig, axes = plt.subplots(nrows = 1, ncols = num_columns, figsize = (4 * num_columns, 5), sharey = True)

        # Define a list of colors to cycle through for the features
        colors = self.palette

        # Plot each feature in a separate subplot
        for i, (column, ax) in enumerate(zip(df.columns, axes)):
            # Plot the feature with a unique color
            ax.plot(df[column], label = column, color = colors[i % len(colors)])

            # Calculate mean and median
            mean_val = df[column].mean()
            median_val = df[column].median()

            # Plot mean and median as horizontal lines
            ax.axhline(mean_val, color = 'red', linestyle = '--', label = f'Mean: {mean_val:.2f}')
            ax.axhline(median_val, color = 'green', linestyle = '--', label = f'Median: {median_val:.2f}')

            # Annotate mean and median
            ax.text(len(df) - 1, mean_val, f'Mean: {mean_val:.2f}', color = 'red', verticalalignment = 'center', fontsize = 8)
            ax.text(len(df) - 1, median_val, f'Median: {median_val:.2f}', color = 'green', verticalalignment = 'center',
                    fontsize = 8)

            # Title and legend for each subplot
            ax.set_title(f'Feature: {column} for {df_name}')
            ax.legend()

        # Adjust layout
        plt.tight_layout()
        plt.show()
