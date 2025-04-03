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
    'visualize_results',
    'Interpreter',
    'PhysicaParser',
    'ClassicalSimulator',
    'QuantumSimulator', 
    'Visualizer'
]

def run_simulation(code, visualize=False):
    """
    Run a physics simulation from PhysicaLang code.
    
    Args:
        code (str): PhysicaLang code to execute
        visualize (bool): Whether to generate visualizations
        
    Returns:
        dict: Dictionary containing simulation results
    """
    interpreter = Interpreter()
    results = interpreter.interpret(code)
    return results

def parse_code(code):
    """
    Parse PhysicaLang code without running a simulation.
    
    Args:
        code (str): PhysicaLang code to parse
        
    Returns:
        object: The AST representation of the code
    """
    parser = PhysicaParser()
    ast_builder = ASTBuilder()
    parse_tree = parser.parse(code)
    program = ast_builder.transform(parse_tree)
    return program

def visualize_results(results, format='html'):
    """
    Create visualizations from simulation results.
    
    Args:
        results (dict): Simulation results from run_simulation()
        format (str): Output format ('html', 'data', or 'matplotlib')
        
    Returns:
        object: Visualization data in requested format
    """
    visualizer = Visualizer()
    
    if "simulation_results" not in results:
        raise ValueError("Invalid results format. Must contain 'simulation_results'.")
    
    # For HTML format, generate the full report
    if format.lower() == 'html':
        from .main import PhysicaLangCLI
        cli = PhysicaLangCLI()
        return cli._generate_html_report(results)
    
    # Other formats to be implemented
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'html'.") 