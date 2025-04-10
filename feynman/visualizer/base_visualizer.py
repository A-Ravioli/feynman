import dash
from dash import dcc, html
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import numpy as np

class BaseVisualizer:
    """Base class for Feynman visualizers using Dash."""
    
    def __init__(self, sim_data):
        """Initialize the visualizer with simulation data."""
        if not self._validate_data(sim_data):
            raise ValueError("Invalid simulation data structure provided.")
            
        self.sim_data = sim_data
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.app.title = "Feynman Simulation Visualizer"
        
        # Prepare data structures
        self.time_points = sim_data.get("time_points", np.array([]))
        self.entities = sim_data.get("entities", {})
        self.entity_names = list(self.entities.keys())
        self.num_steps = len(self.time_points)
        # Store interactions if present (optional)
        self.interactions = sim_data.get("interactions", {})
        self.simulation_parameters = sim_data.get("simulation_parameters", {})

    def _validate_data(self, data):
        """Basic validation of the input simulation data structure."""
        if not isinstance(data, dict):
            print("Validation Error: Data must be a dictionary.")
            return False
        if "time_points" not in data or not isinstance(data["time_points"], np.ndarray):
            print("Validation Error: Missing or invalid 'time_points' (must be NumPy array).")
            return False
        if "entities" not in data or not isinstance(data["entities"], dict):
            print("Validation Error: Missing or invalid 'entities' (must be dictionary).")
            return False
        # Interactions key is optional, but if present, should be a list
        if "interactions" in data and not isinstance(data["interactions"], list):
            print(f"Validation Error: Invalid 'interactions' (type: {type(data.get('interactions'))}, expected list).")
            return False
        for name, entity_data in data["entities"].items():
            if not isinstance(entity_data, dict) or \
               "initial_properties" not in entity_data or \
               "time_series" not in entity_data:
                print(f"Validation Error: Invalid structure for entity '{name}'.")
                return False
            if not isinstance(entity_data["time_series"], dict):
                 print(f"Validation Error: Invalid 'time_series' for entity '{name}'.")
                 return False
        # Add more specific checks if needed (e.g., presence of 'positions')
        return True

    def _create_layout(self):
        """Create the basic Dash layout. Should be implemented by subclasses."""
        return html.Div([
            html.H1("Feynman Simulation Visualizer"),
            html.Hr(),
            html.Div(id='visualization-content', children=[
                html.P("Select visualization type and entities from controls.")
            ]),
            html.Hr(),
            # Controls (e.g., time slider, entity selector) will go here
            self._create_controls()
        ])

    def _create_controls(self):
        """Create common control elements. Can be extended by subclasses."""
        if self.num_steps <= 1:
            return html.Div([html.P("Not enough time steps for animation.")])
        
        return html.Div([
            html.Label("Time Step:"),
            dcc.Slider(
                id='time-slider',
                min=0,
                max=self.num_steps - 1,
                value=0,
                step=1,
                marks={i: f'{self.time_points[i]:.2f}' for i in range(0, self.num_steps, max(1, self.num_steps // 10))},
                tooltip={'placement': 'bottom', 'always_visible': False}
            ),
            # Placeholder for play/pause button
            html.Button('Play/Pause', id='play-pause-button', n_clicks=0),
            dcc.Interval(id='animation-interval', interval=100, max_intervals=0), # Initially paused
        ])

    def _register_callbacks(self):
        """Register basic callbacks. Should be extended by subclasses."""
        
        # Callback to handle play/pause
        @self.app.callback(
            Output('animation-interval', 'max_intervals'),
            Input('play-pause-button', 'n_clicks'),
            State('animation-interval', 'max_intervals')
        )
        def toggle_animation(n_clicks, max_intervals):
            if n_clicks > 0:
                # If playing (max_intervals == -1), pause (max_intervals = 0)
                # If paused (max_intervals == 0), play (max_intervals = -1)
                return 0 if max_intervals == -1 else -1
            return max_intervals # Initial state or no clicks yet

        # Callback to advance slider during animation
        @self.app.callback(
            Output('time-slider', 'value'),
            Input('animation-interval', 'n_intervals'),
            State('time-slider', 'value'),
            State('time-slider', 'max'),
        )
        def advance_slider(n_intervals, current_value, max_value):
            if n_intervals is None or current_value is None:
                return dash.no_update
            
            new_value = (current_value + 1) % (max_value + 1)
            return new_value

    def run_server(self, *args, **kwargs):
        """Run the Dash server."""
        self.app.run(*args, **kwargs) 