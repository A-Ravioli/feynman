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
from .visualizer.visualizer import Visualizer

# Public API
__all__ = [
    'run_simulation',
    'parse_code',
    'launch_visualizer',
    'Interpreter',
    'PhysicaParser',
    'ClassicalSimulator',
    'QuantumSimulator',
    'Visualizer'
]

def run_simulation(code, visualize=False):
    """
    Run a physics simulation from Feynman code.

    Args:
        code (str): Feynman code (.fyn) to execute.
        visualize (bool): If True, launch the interactive visualizer after simulation.
                          (Note: This will block until the visualizer is closed).

    Returns:
        dict: Dictionary containing simulation results.
    """
    interpreter = Interpreter()
    results = interpreter.interpret(code)

    if visualize:
        launch_visualizer(results)
        # Note: If visualize is True, the function will block here until
        # the Dash server is stopped. Consider running in a thread if needed.

    return results # Return results regardless of visualization

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
    Launch the interactive Dash visualizer with the given simulation results.

    Args:
        results (dict): Simulation results dictionary (e.g., from run_simulation()).
                        This function assumes results contain data in the expected format
                        (including NumPy arrays if applicable, not just lists from JSON).
    """
    # No need to check for simulation_results key here, Visualizer handles data prep
    if not isinstance(results, dict) or not results.get('entities'):
         print("Warning: Invalid or empty results provided to visualizer. Nothing to display.")
         return
         
    print("Launching interactive visualizer...")
    try:
        visualizer = Visualizer(results) # Instantiates the Dash app
        # The run_server method blocks until the server is stopped
        visualizer.run_server(debug=False) # Set debug=True for development
        print("Visualizer closed.")
    except Exception as e:
         print(f"Error launching visualizer: {e}")
         import traceback
         traceback.print_exc()

# Remove old visualize_results function
# def visualize_results(...): 