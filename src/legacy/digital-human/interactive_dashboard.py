# interactive_dashboard.py

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

class DashboardApp:
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.data = self.load_sample_data()

    def load_sample_data(self):
        # Sample data for demonstration purposes
        data = {
            "Agent": ["Agent1", "Agent2", "Agent3"],
            "Performance": [80, 90, 85],
            "Category": ["A", "B", "C"],
            "Values": [10, 20, 30]
        }
        return pd.DataFrame(data)

    def generate_agent_plot(self):
        fig = px.line(self.data, x="Agent", y="Performance", title="Agent Performance")
        return fig

    def generate_data_plot(self):
        fig = px.bar(self.data, x="Category", y="Values", title="Category Values")
        return fig

    def create_layout(self):
        self.app.layout = html.Div(children=[
            html.H1(children='Interactive Dashboard'),
            dcc.Tabs(id="tabs", children=[
                dcc.Tab(label='Dashboard 1', children=[
                    html.Div([
                        html.H2('Interactive Dashboard 1'),
                        dcc.Graph(
                            id='example-graph',
                            figure=self.generate_data_plot()
                        )
                    ])
                ]),
                dcc.Tab(label='Dashboard 2', children=[
                    html.Div([
                        html.H2('Interactive Dashboard 2'),
                        dcc.Graph(
                            id='agent-graph',
                            figure=self.generate_agent_plot()
                        )
                    ])
                ])
            ])
        ])

    def run(self):
        self.create_layout()
        self.app.run_server(debug=True)

if __name__ == '__main__':
    app = DashboardApp()
    app.run()
