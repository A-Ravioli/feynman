import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np

from .base_visualizer import BaseVisualizer

class Visualizer3D(BaseVisualizer):
    """Concrete visualizer for 3D animations."""

    def _create_layout(self):
        """Override layout to include the 3D plot area."""
        return html.Div([
            html.H1("Feynman Simulation Visualizer - 3D Animation"),
            html.Hr(),
            html.Div(id='visualization-content', children=[
                dcc.Graph(id='3d-animation-graph', style={'height': '700px'}) # Define graph area
            ]),
            html.Hr(),
            self._create_controls() # Use base controls (slider, play/pause)
        ])

    def _register_callbacks(self):
        """Register callbacks specific to the 3D visualizer."""
        super()._register_callbacks() # Register base callbacks

        # Callback to update the 3D graph based on the time slider
        @self.app.callback(
            Output('3d-animation-graph', 'figure'),
            Input('time-slider', 'value')
        )
        def update_3d_graph(time_index):
            if time_index is None or not self.entities:
                return go.Figure() # Return empty figure if no data

            return self._create_3d_figure(time_index)

    def _create_3d_figure(self, time_index):
        """Creates the Plotly figure for the 3D visualization at a specific time index."""
        min_coords = [np.inf, np.inf, np.inf]
        max_coords = [-np.inf, -np.inf, -np.inf]

        # First pass: determine plot boundaries across all time steps
        valid_entities_found = False
        for name, data in self.entities.items():
            ts = data.get('time_series', {})
            pos = ts.get('positions', None)
            if pos is not None and len(pos.shape) == 2 and pos.shape[1] == 3:
                 valid_entities_found = True
                 min_coords[0] = min(min_coords[0], np.min(pos[:, 0]))
                 min_coords[1] = min(min_coords[1], np.min(pos[:, 1]))
                 min_coords[2] = min(min_coords[2], np.min(pos[:, 2]))
                 max_coords[0] = max(max_coords[0], np.max(pos[:, 0]))
                 max_coords[1] = max(max_coords[1], np.max(pos[:, 1]))
                 max_coords[2] = max(max_coords[2], np.max(pos[:, 2]))

        if not valid_entities_found:
            # Return an empty figure with a note if no valid 3D data exists
            layout = go.Layout(title="No valid 3D position data found for any entity.")
            return go.Figure(layout=layout)

        # Calculate ranges and padding, ensuring roughly cubic aspect ratio
        ranges = []
        max_range_diff = 0
        for i in range(3):
            diff = max_coords[i] - min_coords[i]
            padding = diff * 0.1 if diff > 1e-6 else 1
            ranges.append([min_coords[i] - padding, max_coords[i] + padding])
            max_range_diff = max(max_range_diff, ranges[i][1] - ranges[i][0])

        # Center ranges based on the largest difference
        final_ranges = []
        for r in ranges:
             center = (r[0] + r[1]) / 2
             final_ranges.append([center - max_range_diff / 2, center + max_range_diff / 2])

        x_range, y_range, z_range = final_ranges

        # Second pass: add traces for the current time step
        traces = []
        for name, data in self.entities.items():
            props = data.get('initial_properties', {})
            ts = data.get('time_series', {})
            pos_all = ts.get('positions', None)

            # Check if this entity has valid 3D data
            if pos_all is not None and len(pos_all.shape) == 2 and pos_all.shape[1] == 3 and time_index < len(pos_all):
                current_pos = pos_all[time_index]
                color = props.get('color', 'blue')
                size_prop = props.get('size', 1)
                try:
                     if isinstance(size_prop, (list, np.ndarray)):
                         marker_size = float(size_prop[0]) * 5 # Scale size differently for 3D
                     else:
                          marker_size = float(size_prop) * 5
                except (ValueError, TypeError, IndexError):
                     marker_size = 5 # Default size for 3D

                marker_size = max(2, min(marker_size, 50)) # Clamp size

                trace = go.Scatter3d(
                    x=[current_pos[0]],
                    y=[current_pos[1]],
                    z=[current_pos[2]],
                    mode='markers+text',
                    marker=dict(color=color, size=marker_size, symbol='circle'),
                    name=name,
                    text=[name],
                    textposition="top center"
                )
                traces.append(trace)

        layout = go.Layout(
            title=f"3D Simulation State at Time: {self.time_points[time_index]:.2f}",
            scene=dict(
                xaxis=dict(title='X', range=x_range),
                yaxis=dict(title='Y', range=y_range),
                zaxis=dict(title='Z', range=z_range),
                aspectmode='cube'  # Ensure equal aspect ratio
            ),
            showlegend=True,
            margin=dict(l=10, r=10, t=80, b=10)
        )

        return go.Figure(data=traces, layout=layout) 