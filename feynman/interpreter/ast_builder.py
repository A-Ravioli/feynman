from lark import Transformer, Token
from typing import Dict, List, Any, Optional, Union

from .ast import (
    Program, Model, Object, Atom, Field, 
    Interaction, Simulate, Visualize
)

class ASTBuilder(Transformer):
    def __init__(self):
        super().__init__()
        self.program = Program()
    
    def start(self, items):
        return self.program
    
    def model(self, items):
        name = str(items[0])
        props = self._collect_properties(items[1])
        model = Model(name=name, properties=props)
        self.program.add_model(model)
        return model
    
    def entity(self, items):
        entity_type = str(items[0])
        name = str(items[1])
        props = self._collect_properties(items[2])
        
        if entity_type == "object":
            obj = Object(name=name, properties=props)
            self.program.add_object(obj)
            return obj
        elif entity_type == "atom":
            atom = Atom(name=name, properties=props)
            self.program.add_atom(atom)
            return atom
    
    def field(self, items):
        name = str(items[0])
        props = self._collect_properties(items[1])
        field = Field(name=name, properties=props)
        self.program.add_field(field)
        return field
    
    def interaction(self, items):
        source = str(items[0])
        target = str(items[1])
        props = self._collect_properties(items[2])
        interaction = Interaction(source=source, target=target, properties=props)
        self.program.add_interaction(interaction)
        return interaction
    
    def simulate(self, items):
        model_name = str(items[0])
        simulate = Simulate(model_name=model_name)
        self.program.add_simulate(simulate)
        return simulate
    
    def visualize(self, items):
        entity_name = str(items[0])
        visualization_type = None
        target_name = None
        
        # Check if we have additional items for visualization_type or target_name
        if len(items) > 1:
            # The second item could be either visualization_type or target_name
            # In the grammar, this would need to be handled based on the token type or keywords
            # For now, assuming a simple implementation where:
            # - If 3 items: entity_name, visualization_type, target_name
            # - If 2 items: entity_name, target_name
            if len(items) > 2:
                visualization_type = str(items[1])
                target_name = str(items[2])
            else:
                target_name = str(items[1])
        
        visualize = Visualize(entity_name=entity_name, visualization_type=visualization_type, target_name=target_name)
        self.program.add_visualize(visualize)
        return visualize
    
    def model_props(self, items):
        return items
    
    def entity_props(self, items):
        return items
    
    def field_props(self, items):
        return items
    
    def interaction_props(self, items):
        return items
    
    def model_prop(self, items):
        return (str(items[0]), items[1])
    
    def entity_prop(self, items):
        return (str(items[0]), items[1])
    
    def field_prop(self, items):
        return (str(items[0]), items[1])
    
    def interaction_prop(self, items):
        return (str(items[0]), items[1])
    
    def value(self, items):
        return items[0]
    
    def string(self, items):
        return str(items[0])[1:-1]  # Remove quotes
    
    def number(self, items):
        return items[0]
    
    def fraction(self, items):
        numerator = int(items[0])
        denominator = int(items[1])
        return float(numerator) / float(denominator)
    
    def boolean(self, items):
        return items[0].lower() == "true"
    
    def array(self, items):
        return list(items)
    
    def range(self, items):
        start = int(items[0])
        end = int(items[1])
        return f"{start}..{end}"
    
    def function_call(self, items):
        name = str(items[0])
        args = items[1:] if len(items) > 1 else []
        return {"function": name, "args": args}
    
    def dict_block(self, items):
        result = {}
        for item in items:
            key, value = item
            result[key] = value
        return result
    
    def dict_items(self, items):
        return items
    
    def NAME(self, token):
        return str(token)
    
    def INT(self, token):
        return int(token)
    
    def FLOAT(self, token):
        return float(token)
    
    def _collect_properties(self, props_list) -> Dict[str, Any]:
        """Helper to collect properties from a list of (name, value) tuples"""
        result = {}
        for prop in props_list:
            name, value = prop
            result[name] = value
        return result 