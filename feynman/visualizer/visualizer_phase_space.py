import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np

from .base_visualizer import BaseVisualizer

class VisualizerPhaseSpace(BaseVisualizer):
    """Visualizer for plotting phase space trajectories."""

    def _create_layout(self):
        """Override layout for phase space plots with entity/axis selectors."""
        if not self.entity_names:
             return html.Div([html.H1("Phase Space Visualizer"), html.P("No entities found in simulation data.")])

        # Define options for axes (position/velocity components)
        axis_options = [
            {'label': 'X Position (x)', 'value': 'pos_0'},
            {'label': 'Y Position (y)', 'value': 'pos_1'},
            {'label': 'Z Position (z)', 'value': 'pos_2'},
            {'label': 'X Velocity (vx)', 'value': 'vel_0'},
            {'label': 'Y Velocity (vy)', 'value': 'vel_1'},
            {'label': 'Z Velocity (vz)', 'value': 'vel_2'}
        ]

        return html.Div([
            html.H1("Feynman Simulation Visualizer - Phase Space"),
            html.Hr(),
            html.Div([
                html.Div([
                    html.Label("Select Entity:"),
                    dcc.Dropdown(
                        id='phase-space-entity-selector',
                        options=[{'label': name, 'value': name} for name in self.entity_names],
                        value=self.entity_names[0] # Default to first entity
                    ),
                ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),

                html.Div([
                    html.Label("Select X-Axis:"),
                    dcc.Dropdown(
                        id='phase-space-x-axis-selector',
                        options=axis_options,
                        value='pos_0' # Default to X position
                    ),
                ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),

                html.Div([
                    html.Label("Select Y-Axis:"),
                    dcc.Dropdown(
                        id='phase-space-y-axis-selector',
                        options=axis_options,
                        value='vel_0' # Default to X velocity
                    ),
                ], style={'width': '30%', 'display': 'inline-block'}),
            ]),
            html.Div(id='visualization-content', children=[
                dcc.Graph(id='phase-space-graph', style={'height': '600px'})
            ]),
            html.Hr(),
            self._create_controls() # Use base controls (slider, play/pause)
        ])

    def _register_callbacks(self):
        """Register callbacks for phase space plot updates."""
        super()._register_callbacks() # Register base callbacks

        @self.app.callback(
            Output('phase-space-graph', 'figure'),
            [
                Input('time-slider', 'value'),
                Input('phase-space-entity-selector', 'value'),
                Input('phase-space-x-axis-selector', 'value'),
                Input('phase-space-y-axis-selector', 'value')
            ]
        )
        def update_phase_space_graph(time_index, selected_entity, x_axis_key, y_axis_key):
            if time_index is None or not selected_entity or not x_axis_key or not y_axis_key:
                return go.Figure() # Return empty if inputs invalid

            if selected_entity not in self.entities:
                 return go.Figure(layout=go.Layout(title=f"Entity '{selected_entity}' not found."))

            return self._create_phase_space_figure(time_index, selected_entity, x_axis_key, y_axis_key)

    def _get_axis_data(self, entity_data, axis_key):
        """Helper to extract data based on axis key (e.g., 'pos_0', 'vel_1')."""
        ts = entity_data.get('time_series', {})
        
        # Add basic error handling for invalid axis key format
        try:
            data_type, index_str = axis_key.split('_')
            index = int(index_str)
        except (ValueError, AttributeError):
             print(f"Warning: Invalid axis key format: {axis_key}")
             return None

        data_array = None
        if data_type == 'pos':
            data_array = ts.get('positions')
        elif data_type == 'vel':
            data_array = ts.get('velocities')
        # Can be extended for energy, etc.

        if data_array is not None and isinstance(data_array, np.ndarray) and data_array.ndim == 2 and data_array.shape[1] > index:
            return data_array[:, index]
        else:
            # Add more specific warning
            if data_array is None:
                print(f"Warning: Data type '{data_type}' not found in time_series for entity.")
            elif not isinstance(data_array, np.ndarray):
                 print(f"Warning: Data for '{data_type}' is not a NumPy array.")
            elif data_array.ndim != 2:
                 print(f"Warning: Data for '{data_type}' does not have 2 dimensions (shape: {data_array.shape}).")
            elif data_array.shape[1] <= index:
                 print(f"Warning: Index {index} out of bounds for '{data_type}' data (shape: {data_array.shape}).")
            return None # Data not available or invalid dimension

    def _get_axis_label(self, axis_key):
         """Helper to get a human-readable label from the axis key."""
         labels = {
            'pos_0': 'X Position', 'pos_1': 'Y Position', 'pos_2': 'Z Position',
            'vel_0': 'X Velocity', 'vel_1': 'Y Velocity', 'vel_2': 'Z Velocity'
         }
         # Handle potentially invalid keys gracefully
         return labels.get(str(axis_key), str(axis_key))

    def _create_phase_space_figure(self, time_index, entity_name, x_axis_key, y_axis_key):
        """Creates the Plotly figure for the phase space plot."""
        entity_data = self.entities[entity_name]
        props = entity_data.get('initial_properties', {})
        color = props.get('color', 'blue')

        x_data = self._get_axis_data(entity_data, x_axis_key)
        y_data = self._get_axis_data(entity_data, y_axis_key)

        x_label = self._get_axis_label(x_axis_key)
        y_label = self._get_axis_label(y_axis_key)

        # Check data validity after trying to fetch it
        if x_data is None or y_data is None or time_index >= len(x_data):
            title = f"Data not available for axes ({x_label}, {y_label}) for {entity_name}"
            if x_data is not None and time_index >= len(x_data):
                 title += f" at time index {time_index}."
            else:
                 title += "."
            return go.Figure(layout=go.Layout(title=title))

        # Plot trajectory up to current time_index
        x_trajectory = x_data[:time_index+1]
        y_trajectory = y_data[:time_index+1]

        # Get current point
        current_x = x_data[time_index]
        current_y = y_data[time_index]

        fig = go.Figure()

        # Add trajectory line
        fig.add_trace(go.Scatter(
            x=x_trajectory,
            y=y_trajectory,
            mode='lines',
            line=dict(color=color, width=2),
            name=f'{entity_name} Trajectory'
        ))

        # Add current position marker
        fig.add_trace(go.Scatter(
            x=[current_x],
            y=[current_y],
            mode='markers',
            marker=dict(color=color, size=12, symbol='circle'),
            name=f'{entity_name} Current State'
        ))

        layout = go.Layout(
            title=f"Phase Space: {entity_name} ({y_label} vs {x_label}) at Time {self.time_points[time_index]:.2f}",
            xaxis=dict(title=x_label), # Use generated labels
            yaxis=dict(title=y_label),
            showlegend=True,
            margin=dict(l=40, r=40, t=80, b=40)
            # Set axis ranges based on the full trajectory for stability
            # Add a small padding to the ranges
            # xaxis_range=[np.min(x_data) - (np.max(x_data) - np.min(x_data))*0.05, np.max(x_data) + (np.max(x_data) - np.min(x_data))*0.05],
            # yaxis_range=[np.min(y_data) - (np.max(y_data) - np.min(y_data))*0.05, np.max(y_data) + (np.max(y_data) - np.min(y_data))*0.05]
        )
        
        # Set axis ranges dynamically based on full trajectory data
        try:
            x_min, x_max = np.min(x_data), np.max(x_data)
            y_min, y_max = np.min(y_data), np.max(y_data)
            x_pad = (x_max - x_min) * 0.05 if (x_max - x_min) > 1e-6 else 0.1
            y_pad = (y_max - y_min) * 0.05 if (y_max - y_min) > 1e-6 else 0.1
            fig.update_layout(xaxis_range=[x_min - x_pad, x_max + x_pad])
            fig.update_layout(yaxis_range=[y_min - y_pad, y_max + y_pad])
        except ValueError:
             # Handle cases where data might be empty or invalid for min/max
             print("Warning: Could not determine dynamic axis ranges for phase space plot.")

        return fig