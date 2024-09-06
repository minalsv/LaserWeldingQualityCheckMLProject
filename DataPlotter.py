"""
The data plotter allows to plot the data by setting various color palettes and type of graphs.
"""
import matplotlib
matplotlib.use('TkAgg')  # Choose an interactive backend
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

