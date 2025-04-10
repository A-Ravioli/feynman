"""
API for Feynman - Universal Physics DSL

This module provides programmatic access to Feynman's functionality,
allowing developers to integrate physics simulations into their applications.
"""

from .interpreter.interpreter import Interpreter
from .parser.parser import PhysicaParser
from .interpreter.ast_builder import ASTBuilder
from .simulator.classical_simulator import ClassicalSimulator
from .simulator.quantum_simulator import QuantumSimulator
# Remove old visualizer import
# from .visualizer.visualizer_2d import Visualizer2D 
# Import the main visualizer
from .visualizer.main_visualizer import FeynmanVisualizer

# Public API
__all__ = [
    'run_simulation',
    'parse_code',
    'launch_visualizer',
    'Interpreter',
    'PhysicaParser',
    'ClassicalSimulator',
    'QuantumSimulator',
    # 'Visualizer2D' # Remove specific visualizer
    'FeynmanVisualizer' # Add main visualizer
]

def run_simulation(code, visualize=False):
    """
    Run a physics simulation from Feynman code.

    Args:
        code (str): Feynman code (.fyn) to execute.
        visualize (bool): If True, launch the interactive visualizer after simulation.
                          (Note: This will block until the visualizer is closed).

    Returns:
        dict: Dictionary containing structured simulation results.
    """
    interpreter = Interpreter()
    structured_results = interpreter.interpret(code)

    if visualize:
        launch_visualizer(structured_results)
        # Note: If visualize is True, the function will block here until
        # the Dash server is stopped. Consider running in a thread if needed.

    return structured_results # Return the structured results

def parse_code(code):
    """
    Parse Feynman code without running a simulation.

    Args:
        code (str): Feynman code to parse

    Returns:
        object: The AST representation of the code (specific type depends on ASTBuilder)
    """
    parser = PhysicaParser()
    ast_builder = ASTBuilder()
    parse_tree = parser.parse(code)
    program = ast_builder.transform(parse_tree)
    return program

def launch_visualizer(results):
    """
    Launch the main interactive Dash visualizer with tabs for different plots.
    
    Args:
        results (dict): Structured simulation results dictionary (from run_simulation()).
                        Expected format:
                        {
                            "time_points": np.array([...]),
                            "entities": {
                                "entity_name": {
                                    "initial_properties": {...},
                                    "time_series": {...}
                                }, ...
                            },
                            "simulation_parameters": {...}
                        }
    """
    if not isinstance(results, dict) or \
       "time_points" not in results or \
       "entities" not in results or \
       not isinstance(results["entities"], dict):
         print("Warning: Invalid or improperly structured results provided to visualizer. Nothing to display.")
         return
         
    if not results["entities"]:
        print("Warning: Simulation results contain no entities to visualize.")
        return

    print("Launching main interactive visualizer...")
    # print(f"Data to visualize: {results}") # Optional: Print data for debugging
    try:
        # Instantiate and run the main FeynmanVisualizer
        visualizer = FeynmanVisualizer(results) # Instantiates the Dash app with tabs
        # The run_server method blocks until the server is stopped
        visualizer.run_server(debug=False) # Set debug=True for development
        
        # Remove placeholder prints
        # print("--- Visualizer Placeholder ---")
        # print(f"  Time points: {len(results.get('time_points', []))}")
        # print(f"  Entities: {list(results.get('entities', {}).keys())}")
        # print(f"  Sim Params: {results.get('simulation_parameters', {})}")
        # print("--- Visualization would run here (blocking) ---")
        
        print("Visualizer closed.")
    except Exception as e:
         print(f"Error launching visualizer: {e}")
         import traceback
         traceback.print_exc()

# Remove old visualize_results function
# def visualize_results(...): 