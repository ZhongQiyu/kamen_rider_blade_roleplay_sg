# visualization.py

import pandas as pd
import plotly.express as px

# VisualizationHelper class
class VisualizationHelper:
    def __init__(self, filepath: str = None):
        self.filepath = filepath
        self.data = None

    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """Load data from a CSV file."""
        if filepath:
            self.filepath = filepath
        if not self.filepath:
            raise ValueError("Filepath must be provided to load data.")
        
        self.data = pd.read_csv(self.filepath)
        return self.data

    def clean_data(self) -> pd.DataFrame:
        """Perform basic data cleaning."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        self.data = self.data.dropna()  # Drop rows with missing values
        self.data = self.data[self.data.select_dtypes(include=['number']).ge(0).all(1)]  # Remove negative values for numeric columns
        return self.data

    def create_bar_plot(self, x_col: str, y_col: str, title: str) -> px.bar:
        """Create a bar plot."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        fig = px.bar(self.data, x=x_col, y=y_col, title=title)
        return fig

    def create_line_plot(self, x_col: str, y_col: str, title: str) -> px.line:
        """Create a line plot."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        fig = px.line(self.data, x=x_col, y=y_col, title=title)
        return fig

    def create_scatter_plot(self, x_col: str, y_col: str, title: str) -> px.scatter:
        """Create a scatter plot."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        fig = px.scatter(self.data, x=x_col, y=y_col, title=title)
        return fig

# DataVisualizer class
class DataVisualizer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.helper = VisualizationHelper(filepath)

    def load_and_clean_data(self):
        self.data = self.helper.load_data()
        self.data = self.helper.clean_data()
    
    def generate_plot(self):
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_and_clean_data() first.")
        
        fig = self.helper.create_bar_plot(x_col="Category", y_col="Values", title="Category Values")
        return fig

# AgentVisualizer class
class AgentVisualizer:
    def __init__(self, data=None):
        """
        Initialize the AgentVisualizer with optional data.
        :param data: Optional initial data as a dictionary.
        """
        if data is None:
            data = {
                "Agent": ["Agent1", "Agent2", "Agent3"],
                "Performance": [80, 90, 85]
            }
        self.df = pd.DataFrame(data)
        self.helper = VisualizationHelper()

    def set_data(self, data):
        """
        Set the data for visualization.
        :param data: Data as a dictionary.
        """
        self.df = pd.DataFrame(data)

    def get_data(self):
        """
        Get the current data.
        :return: Current data as a pandas DataFrame.
        """
        return self.df

    def generate_plot(self, x_col="Agent", y_col="Performance", title="Agent Performance"):
        """
        Generate a line plot for the current data.
        :param x_col: Column name for the x-axis.
        :param y_col: Column name for the y-axis.
        :param title: Title of the plot.
        :return: Plotly figure.
        """
        self.helper.data = self.df
        return self.helper.create_line_plot(x_col=x_col, y_col=y_col, title=title)
