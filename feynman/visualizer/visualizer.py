import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import webbrowser
import threading
import time
from typing import Dict, Any, List, Tuple

class Visualizer:
    """Creates an interactive Dash application for visualizing simulation results."""

    DEFAULT_PORT = 8050

    def __init__(self, results: Dict[str, Any], port=DEFAULT_PORT):
        """
        Initializes the Visualizer with simulation results.

        Args:
            results (Dict[str, Any]): The results dictionary from the simulator.
            port (int): The port to run the Dash server on.
        """
        self.results = results
        self.port = port
        self.app = dash.Dash(__name__)
        self.entity_data = self._prepare_entity_data()
        self.time_points = results.get('time_points', np.array([0]))
        self.min_time = self.time_points[0]
        self.max_time = self.time_points[-1]
        self.time_step = self.time_points[1] - self.time_points[0] if len(self.time_points) > 1 else 1
        self._setup_layout()
        self._setup_callbacks()

    def _prepare_entity_data(self) -> Dict[str, Dict]:
        """Processes the results dictionary into a format suitable for plotting."""
        data = {}
        entities = self.results.get('entities', {})
        for name, entity_result in entities.items():
            entity_type = entity_result.get('type', 'unknown')
            entity_props = self.results.get('simulation_params', {}).get('entities', {}).get(name, {}).get('properties', {})
            mass = entity_result.get('mass', entity_props.get('mass', 1.0))
            
            initial_vis_props = entity_props.get('visualization', {})
            color = initial_vis_props.get('color', None) # Allow specific colors
            size = initial_vis_props.get('size', 5) # Default marker size

            plot_data = {
                'type': entity_type,
                'color': color,
                'size': size,
                'observables': {}
            }

            if entity_type == 'object':
                pos = entity_result.get('positions') # Expected shape: (time, dims)
                if pos is not None:
                    plot_data['positions'] = pos
                    plot_data['observables']['Kinetic Energy'] = entity_result.get('kinetic_energy')
                    # Add other classical observables if available
            elif entity_type == 'atom':
                pos = entity_result.get('expected_position') # Expected shape: (time, dims)
                if pos is not None:
                    plot_data['positions'] = pos
                    plot_data['observables']['Expected Energy'] = entity_result.get('expected_energy')
                    plot_data['observables']['Expected Momentum X'] = entity_result.get('expected_momentum', np.zeros_like(pos))[:, 0]
                    if pos.shape[1] > 1:
                         plot_data['observables']['Expected Momentum Y'] = entity_result.get('expected_momentum', np.zeros_like(pos))[:, 1]
                    if pos.shape[1] > 2:
                         plot_data['observables']['Expected Momentum Z'] = entity_result.get('expected_momentum', np.zeros_like(pos))[:, 2]
                # Could add probability density plots later if needed

            # Filter out observables with None values
            plot_data['observables'] = {k: v for k, v in plot_data['observables'].items() if v is not None}

            # Only require position data to include the entity in the visualization
            if 'positions' in plot_data:
                 data[name] = plot_data
            else:
                 print(f"Warning: Skipping entity '{name}' from visualization due to missing position data.")
                 
        return data

    def _setup_layout(self):
        """Defines the layout of the Dash application."""
        if not self.entity_data:
             self.app.layout = html.Div([html.H1("No valid entities found to visualize.")])
             return
             
        entity_names = list(self.entity_data.keys())
        initial_entity = entity_names[0]
        observable_options = [
            {'label': obs_name, 'value': obs_name}
            for obs_name in self.entity_data[initial_entity]['observables'].keys()
        ]

        self.app.layout = html.Div([
            html.H1("Feynman Simulation Visualizer", style={'textAlign': 'center'}),
            html.Div([
                # Left Side: 3D Plot + Controls
                html.Div([
                    dcc.Graph(id='3d-trajectory-plot', style={'height': '60vh'}), # Adjust height as needed
                    html.Div([
                        html.Button('Play/Pause', id='play-pause-button', n_clicks=0),
                        dcc.Slider(
                            id='time-slider',
                            min=0, 
                            max=len(self.time_points) - 1,
                            value=0,
                            step=1,
                            marks=None, # Disable marks for performance with many steps
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '20px'})
                ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                # Right Side: Entity Selector + Observable Plots
                html.Div([
                    html.Label("Select Entity:"),
                    dcc.Dropdown(
                        id='entity-selector',
                        options=[{'label': name, 'value': name} for name in entity_names],
                        value=initial_entity,
                        clearable=False
                    ),
                    html.Label("Select Observable:", style={'marginTop': '20px'}),
                    dcc.Dropdown(
                        id='observable-selector',
                        options=observable_options,
                        value=observable_options[0]['value'] if observable_options else None,
                        clearable=False
                    ),
                    dcc.Graph(id='observable-plot', style={'height': '40vh'})
                ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '3%', 'verticalAlign': 'top'})
            ]),
            dcc.Interval(id='animation-interval', interval=50, max_intervals=0) # Initially paused
        ])

    def _setup_callbacks(self):
        """Defines the callbacks for interactivity."""
        if not self.entity_data:
             return

        # Update 3D plot and time slider based on interval or manual slide
        @self.app.callback(
            Output('3d-trajectory-plot', 'figure'),
            Output('time-slider', 'value'),
            Input('time-slider', 'value'),
            Input('animation-interval', 'n_intervals')
        )
        def update_3d_plot(slider_value, n_intervals):
            ctx = dash.callback_context
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if trigger_id == 'animation-interval':
                current_frame_index = (slider_value + 1) % len(self.time_points)
            else: # Manual slide
                current_frame_index = slider_value

            fig = go.Figure()
            traces = []
            max_range = 0

            # Add traces for full trajectories
            for i, (name, data) in enumerate(self.entity_data.items()):
                pos = data['positions']
                color = data['color'] if data['color'] else f'hsl({(i*60) % 360}, 70%, 50%)'
                dims = pos.shape[1]
                
                # Determine max range for aspect ratio
                current_max = np.max(np.abs(pos))
                if current_max > max_range:
                    max_range = current_max

                if dims == 1:
                    fig.add_trace(go.Scatter3d(
                        x=pos[:, 0],
                        y=np.zeros(len(pos)), 
                        z=np.zeros(len(pos)),
                        mode='lines', 
                        line=dict(color=color, width=2),
                        name=f'{name} (Trajectory)'
                    ))
                    # Add marker for current position
                    fig.add_trace(go.Scatter3d(
                        x=[pos[current_frame_index, 0]],
                        y=[0],
                        z=[0],
                        mode='markers',
                        marker=dict(color=color, size=data['size'], symbol='circle'),
                        name=f'{name} (t={self.time_points[current_frame_index]:.2e})'
                    ))
                elif dims == 2:
                    fig.add_trace(go.Scatter3d(
                        x=pos[:, 0],
                        y=pos[:, 1],
                        z=np.zeros(len(pos)),
                        mode='lines',
                        line=dict(color=color, width=2),
                        name=f'{name} (Trajectory)'
                    ))
                    fig.add_trace(go.Scatter3d(
                        x=[pos[current_frame_index, 0]],
                        y=[pos[current_frame_index, 1]],
                        z=[0],
                        mode='markers',
                        marker=dict(color=color, size=data['size'], symbol='circle'),
                        name=f'{name} (t={self.time_points[current_frame_index]:.2e})'
                    ))
                else: # 3D
                    fig.add_trace(go.Scatter3d(
                        x=pos[:, 0],
                        y=pos[:, 1],
                        z=pos[:, 2],
                        mode='lines',
                        line=dict(color=color, width=2),
                        name=f'{name} (Trajectory)'
                    ))
                    fig.add_trace(go.Scatter3d(
                        x=[pos[current_frame_index, 0]],
                        y=[pos[current_frame_index, 1]],
                        z=[pos[current_frame_index, 2]],
                        mode='markers',
                        marker=dict(color=color, size=data['size'], symbol='circle'),
                        name=f'{name} (t={self.time_points[current_frame_index]:.2e})'
                    ))

            # Set layout for 3D plot
            axis_lim = max_range * 1.1 if max_range > 1e-12 else 1.0 # Avoid zero range
            fig.update_layout(
                title=f'Object Trajectories (Time: {self.time_points[current_frame_index]:.3e} s)',
                scene=dict(
                    xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                    xaxis=dict(range=[-axis_lim, axis_lim]),
                    yaxis=dict(range=[-axis_lim, axis_lim]),
                    zaxis=dict(range=[-axis_lim, axis_lim]),
                    aspectmode='cube' # Ensure cubic aspect ratio
                ),
                margin=dict(r=20, l=10, b=10, t=40),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            return fig, current_frame_index

        # Update observable plot based on selection
        @self.app.callback(
            Output('observable-plot', 'figure'),
            Output('observable-selector', 'options'),
            Output('observable-selector', 'value'),
            Input('entity-selector', 'value'),
            Input('observable-selector', 'value'),
            Input('time-slider', 'value') # Link to time slider
        )
        def update_observable_plot(selected_entity, selected_observable, current_frame_index):
            entity_observables = self.entity_data[selected_entity]['observables']
            options = [{'label': obs_name, 'value': obs_name} for obs_name in entity_observables.keys()]
            
            # If the selected observable isn't valid for the new entity, pick the first available one
            valid_observable_names = list(entity_observables.keys())
            if selected_observable not in valid_observable_names:
                selected_observable = valid_observable_names[0] if valid_observable_names else None

            fig = go.Figure()
            if selected_observable and selected_observable in entity_observables:
                y_data = entity_observables[selected_observable]
                fig.add_trace(go.Scatter(
                    x=self.time_points,
                    y=y_data,
                    mode='lines',
                    name=selected_observable
                ))
                # Add a marker for the current time
                current_time = self.time_points[current_frame_index]
                current_value = y_data[current_frame_index]
                fig.add_trace(go.Scatter(
                    x=[current_time],
                    y=[current_value],
                    mode='markers',
                    marker=dict(color='red', size=10, symbol='x'),
                    name='Current Time'
                ))
                fig.update_layout(
                    title=f'{selected_observable} for {selected_entity}',
                    xaxis_title='Time (s)',
                    yaxis_title=selected_observable,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
            else:
                 fig.update_layout(title=f"No observable selected for {selected_entity}")

            return fig, options, selected_observable

        # Play/Pause Animation
        @self.app.callback(
            Output('animation-interval', 'max_intervals'),
            Input('play-pause-button', 'n_clicks'),
            State('animation-interval', 'max_intervals')
        )
        def toggle_animation(n_clicks, max_intervals):
            if n_clicks == 0:
                return 0 # Start paused
            # Toggle between playing (-1) and paused (0)
            return -1 if max_intervals == 0 else 0

    def run_server(self, debug=False):
        """Starts the Dash server."""
        if not self.entity_data:
            print("Cannot start visualization server: No valid data to display.")
            return
            
        url = f"http://127.0.0.1:{self.port}"
        print(f"Starting Dash server on {url} ...")
        
        # Open the browser only if not debugging (to avoid issues with reloader)
        if not debug:
             threading.Timer(1, lambda: webbrowser.open(url)).start() # Open after 1 sec delay
             
        self.app.run(port=self.port, debug=debug)

# Example Usage (for testing directly)
# if __name__ == '__main__':
#     # Create dummy results data
#     num_points = 100
#     dims=3
#     times = np.linspace(0, 10, num_points)
#     pos1 = np.zeros((num_points, dims))
#     pos2 = np.zeros((num_points, dims))
#     pos1[:, 0] = np.sin(times)
#     pos1[:, 1] = np.cos(times)
#     pos1[:, 2] = times / 10
#     pos2[:, 0] = 0.5 * np.sin(2*times)
#     pos2[:, 1] = 0.5 * np.cos(2*times)
#     pos2[:, 2] = 1.0 - times / 10
# 
#     dummy_results = {
#         'time_points': times,
#         'simulation_params': {
#             'entities': { 
#                 'obj1': {'properties': {'mass': 1.0, 'visualization': {'color': 'blue'}}},
#                 'obj2': {'properties': {'mass': 2.0, 'visualization': {'color': 'green'}}}
#             }
#         },
#         'entities': {
#             'obj1': {
#                 'type': 'object',
#                 'positions': pos1,
#                 'kinetic_energy': 0.5 * 1.0 * (np.cos(times)**2 + (-np.sin(times))**2 + (1/10)**2),
#                 'mass': 1.0
#             },
#             'obj2': {
#                 'type': 'object',
#                 'positions': pos2,
#                 'kinetic_energy': 0.5 * 2.0 * ((np.cos(2*times))**2 + (-np.sin(2*times))**2 + (-1/10)**2),
#                 'mass': 2.0
#             }
#         }
#     }
# 
#     visualizer = Visualizer(dummy_results)
#     visualizer.run_server(debug=True) 