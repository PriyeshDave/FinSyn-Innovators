import pandas as pd
import numpy as np
import time
from scipy.stats import norm
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go


class DataAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.llm_client = OpenAI(api_key=self.api_key)
    
    def generate_summary_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for the dataset.
        """
        return data.describe(include='all').transpose()
    


    def generate_column_plot_plotly(self, data: pd.DataFrame, column: str) -> dict:
        """
        Generate multiple interactive plots for a given column using Plotly.
        The plots are generated only for numerical columns.
        """
        plots = {}
        
        # Check if the column is numerical
        if not np.issubdtype(data[column].dtype, np.number):
            # Skip generating plots for non-numeric columns
            return plots
        
        # Gaussian Distribution Plot (for Numerical Columns)
        mean = data[column].mean()
        std = data[column].std()
        x = np.linspace(data[column].min(), data[column].max(), 100)
        y = norm.pdf(x, mean, std)
        
        fig_gaussian = go.Figure()
        fig_gaussian.add_trace(go.Histogram(
            x=data[column],
            histnorm='probability density',
            name='Data Distribution',
            opacity=0.7
        ))
        fig_gaussian.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name='Gaussian Fit',
            line=dict(color='red')
        ))
        fig_gaussian.update_layout(
            title=f"Gaussian Distribution Plot for {column}",
            xaxis_title=column,
            yaxis_title="Density"
        )
        plots['gaussian_distribution_plot'] = fig_gaussian

        # Distribution Plot
        fig_dist = px.histogram(data, x=column, marginal="box", nbins=30, title=f"Distribution Plot for {column}")
        plots['distribution_plot'] = fig_dist
        
        # Box Plot
        fig_box = px.box(data, y=column, title=f"Box Plot for {column}")
        plots['box_plot'] = fig_box
        
        # Outlier Detection Plot
        fig_outliers = go.Figure()
        fig_outliers.add_trace(go.Box(y=data[column], boxpoints='all', name='Outliers'))
        fig_outliers.update_layout(title=f"Outlier Detection for {column}")
        plots['outlier_plot'] = fig_outliers

        return plots



    def generate_column_insight(self, column_name: str, stats: pd.Series) -> str:
        """
        Generate insights for a column using LLM.
        """
        with st.spinner(f"Gererating statistical insights for {column_name}..."):
            progress = st.progress(0)
            for i in range(100):  
                time.sleep(0.02) 
                progress.progress(i + 1)
        prompt = f"""
        Analyze the following statistical summary for the column '{column_name}':
        {stats.to_string()}
        
        Provide a concise summary and key insights based on this information.
        """
        response = self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data analysis assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    


    """
    Payload Structure:

    {
        "summary_statistics": {
            "column_name": {
                "mean": float,
                "std": float,
                "min": float,
                "max": float,
                ... other statistics ...
            },
            ... other columns ...
        },
        "columns": {
            "column_name": {
                "plots": {
                    "plot_name": "plotly_figure_json",
                    ... other plots ...
                },
                "insight": "Generated insight for the column"
            },
            ... other columns ...
        }
    }

    Explanation:
    - "summary_statistics": Dictionary of statistical metrics for each column.
    - "columns": Contains:
        - "plots": JSON representations of Plotly figures for visualization.
        - "insight": A text-based insight generated for each column.
    """
    def show_plots_and_insights(self, dataset):
        summary_stats = self.generate_summary_statistics(dataset)

        payload = {
            "summary_statistics": summary_stats.to_dict(),
            "columns": {}
        }

        for column in dataset.columns:
            column_data = {
                "plots": {},
                "insight": ""
            }

            # Generate and store plots
            column_plots = self.generate_column_plot_plotly(dataset, column)
            if column_plots:
                for plot_name, fig in column_plots.items():
                    column_data["plots"][plot_name] = fig.to_json()
            else:
                column_data["plots"] = "No plots available (non-numeric data)."

            # Generate and store insights
            column_stats = summary_stats.loc[column]
            column_data["insight"] = self.generate_column_insight(column, column_stats)

            payload["columns"][column] = column_data
        payload['statistical_inference'] = payload
        return payload