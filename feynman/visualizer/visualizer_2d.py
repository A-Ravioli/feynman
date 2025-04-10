import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np

from .base_visualizer import BaseVisualizer

class Visualizer2D(BaseVisualizer):
    """Concrete visualizer for 2D animations."""

    def _create_layout(self):
        """Override layout to include the 2D plot area."""
        return html.Div([
            html.H1("Feynman Simulation Visualizer - 2D Animation"),
            html.Hr(),
            html.Div(id='visualization-content', children=[
                dcc.Graph(id='2d-animation-graph', style={'height': '600px'}) # Define graph area
            ]),
            html.Hr(),
            self._create_controls() # Use base controls (slider, play/pause)
        ])

    def _register_callbacks(self):
        """Register callbacks specific to the 2D visualizer."""
        super()._register_callbacks() # Register base callbacks (play/pause, slider advance)

        # Callback to update the 2D graph based on the time slider
        @self.app.callback(
            Output('2d-animation-graph', 'figure'),
            Input('time-slider', 'value')
        )
        def update_2d_graph(time_index):
            if time_index is None or not self.entities:
                return go.Figure() # Return empty figure if no data

            return self._create_2d_figure(time_index)

    def _create_2d_figure(self, time_index):
        """Creates the Plotly figure for the 2D visualization at a specific time index."""
        fig = go.Figure()
        min_coords = [np.inf, np.inf]
        max_coords = [-np.inf, -np.inf]

        # First pass: determine plot boundaries across all time steps
        for name, data in self.entities.items():
            ts = data.get('time_series', {})
            pos = ts.get('positions', None)
            if pos is not None and len(pos.shape) == 2 and pos.shape[1] >= 2:
                 min_coords[0] = min(min_coords[0], np.min(pos[:, 0]))
                 min_coords[1] = min(min_coords[1], np.min(pos[:, 1]))
                 max_coords[0] = max(max_coords[0], np.max(pos[:, 0]))
                 max_coords[1] = max(max_coords[1], np.max(pos[:, 1]))

        # Handle cases with no data or single point
        if np.isinf(min_coords[0]) or np.isinf(max_coords[0]):
             x_range = [-1, 1]
             y_range = [-1, 1]
        else:
            # Add padding to boundaries
            x_range = [min_coords[0], max_coords[0]]
            y_range = [min_coords[1], max_coords[1]]
            x_diff = x_range[1] - x_range[0]
            y_diff = y_range[1] - y_range[0]
            x_padding = x_diff * 0.1 if x_diff > 1e-6 else 1
            y_padding = y_diff * 0.1 if y_diff > 1e-6 else 1
            x_range = [x_range[0] - x_padding, x_range[1] + x_padding]
            y_range = [y_range[0] - y_padding, y_range[1] + y_padding]
            
            # Make aspect ratio roughly equal based on padded ranges
            x_diff_padded = x_range[1] - x_range[0]
            y_diff_padded = y_range[1] - y_range[0]
            if x_diff_padded > y_diff_padded:
                center_y = (y_range[0] + y_range[1]) / 2
                y_range = [center_y - x_diff_padded / 2, center_y + x_diff_padded / 2]
            elif y_diff_padded > x_diff_padded:
                 center_x = (x_range[0] + x_range[1]) / 2
                 x_range = [center_x - y_diff_padded / 2, center_x + y_diff_padded / 2]


        # Second pass: add traces for the current time step
        traces = []
        for name, data in self.entities.items():
            props = data.get('initial_properties', {})
            ts = data.get('time_series', {})
            pos_all = ts.get('positions', None)

            if pos_all is not None and len(pos_all.shape) == 2 and pos_all.shape[1] >= 2 and time_index < len(pos_all):
                current_pos = pos_all[time_index]
                color = props.get('color', 'blue') # Default color
                size_prop = props.get('size', 1) # Get size property
                # Attempt to parse size even if it's a list/vector (e.g., width/height for rectangle)
                try:
                     if isinstance(size_prop, (list, np.ndarray)):
                         marker_size = float(size_prop[0]) * 20 # Use first dimension for size
                     else:
                          marker_size = float(size_prop) * 20
                except (ValueError, TypeError, IndexError):
                     marker_size = 20 # Default if size is invalid
                
                # Clamp marker size to a reasonable range
                marker_size = max(5, min(marker_size, 100))

                trace = go.Scatter(
                    x=[current_pos[0]],
                    y=[current_pos[1]],
                    mode='markers+text',
                    marker=dict(color=color, size=marker_size, symbol='circle'),
                    name=name,
                    text=[name], # Show name next to marker
                    textposition="top center"
                )
                traces.append(trace)

        layout = go.Layout(
            title=f"2D Simulation State at Time: {self.time_points[time_index]:.2f}",
            xaxis=dict(
                title='X Position', 
                range=x_range, 
                # constrain='domain' # Ensure x-axis takes up available horizontal space
            ),
            yaxis=dict(
                title='Y Position', 
                range=y_range, 
                scaleanchor="x", # Anchor y scale to x scale
                scaleratio=1     # Ensure 1:1 aspect ratio (equal unit lengths)
            ),
            showlegend=True,
            margin=dict(l=40, r=40, t=80, b=40) # Adjust margins for better layout
        )

        return go.Figure(data=traces, layout=layout) 