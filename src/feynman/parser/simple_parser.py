import re
from typing import Dict, List, Any, Optional, Tuple
import json

from ..interpreter.ast import (
    Program, Model, Object, Atom, Field, 
    Interaction, Simulate, Visualize
)

class SimpleParser:
    """A simpler parser for PhysicaLang that uses regular expressions"""
    
    def __init__(self):
        self.program = Program()
        
        # Regular expressions for various patterns
        self.model_pattern = re.compile(r'model\s+(\w+)\s*:')
        self.object_pattern = re.compile(r'object\s+(\w+)\s*:')
        self.atom_pattern = re.compile(r'atom\s+(\w+)\s*:')
        self.field_pattern = re.compile(r'field\s+(\w+)\s*:')
        self.interaction_pattern = re.compile(r'interaction\s+(\w+)\s*->\s*(\w+)\s*:')
        self.simulate_pattern = re.compile(r'simulate\s+(\w+)')
        self.visualize_pattern = re.compile(r'visualize\s+(\w+)(?:\s+on\s+(\w+))?')
        self.property_pattern = re.compile(r'(\w+)\s*:\s*(.+)')
        
        # For handling nested structures
        self.indent_size = 4
    
    def parse(self, code: str) -> Program:
        """Parse PhysicaLang code and return AST"""
        self.program = Program()
        
        # Normalize line endings and clean up
        code = code.replace('\r\n', '\n').replace('\r', '\n')
        lines = code.split('\n')
        
        # Strip trailing spaces but maintain leading indentation
        lines = [line.rstrip() for line in lines]
        
        # Remove empty lines and comments
        lines = [line for line in lines if line and not line.strip().startswith('#')]
        
        # Process each block in the code
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Skip empty lines
            if not line.strip():
                i += 1
                continue
            
            # Check for various block types
            match = self.model_pattern.match(line.strip())
            if match:
                name = match.group(1)
                i = self._parse_model(name, lines, i)
                continue
                
            match = self.object_pattern.match(line.strip())
            if match:
                name = match.group(1)
                i = self._parse_object(name, lines, i)
                continue
                
            match = self.atom_pattern.match(line.strip())
            if match:
                name = match.group(1)
                i = self._parse_atom(name, lines, i)
                continue
                
            match = self.field_pattern.match(line.strip())
            if match:
                name = match.group(1)
                i = self._parse_field(name, lines, i)
                continue
                
            match = self.interaction_pattern.match(line.strip())
            if match:
                source = match.group(1)
                target = match.group(2)
                i = self._parse_interaction(source, target, lines, i)
                continue
                
            match = self.simulate_pattern.match(line.strip())
            if match:
                model_name = match.group(1)
                self.program.add_simulate(Simulate(model_name=model_name))
                i += 1
                continue
                
            match = self.visualize_pattern.match(line.strip())
            if match:
                entity_name = match.group(1)
                target_name = match.group(2) if match.group(2) else None
                self.program.add_visualize(Visualize(entity_name=entity_name, target_name=target_name))
                i += 1
                continue
                
            # Skip unrecognized lines
            i += 1
        
        return self.program
    
    def _parse_model(self, name: str, lines: List[str], start_idx: int) -> int:
        """Parse a model block and return the new line index"""
        properties = {}
        
        # Get indentation level for the block
        if start_idx + 1 < len(lines):
            block_indent = self._get_indent_level(lines[start_idx + 1])
            
            # Process properties
            i = start_idx + 1
            while i < len(lines) and self._get_indent_level(lines[i]) >= block_indent:
                line = lines[i].strip()
                prop_match = self.property_pattern.match(line)
                if prop_match:
                    prop_name = prop_match.group(1)
                    prop_value = self._parse_value(prop_match.group(2))
                    properties[prop_name] = prop_value
                i += 1
        else:
            i = start_idx + 1
        
        # Add the model to the program
        model = Model(name=name, properties=properties)
        self.program.add_model(model)
        
        return i
    
    def _parse_object(self, name: str, lines: List[str], start_idx: int) -> int:
        """Parse an object block and return the new line index"""
        properties = {}
        
        # Get indentation level for the block
        if start_idx + 1 < len(lines):
            block_indent = self._get_indent_level(lines[start_idx + 1])
            
            # Process properties
            i = start_idx + 1
            while i < len(lines) and self._get_indent_level(lines[i]) >= block_indent:
                line = lines[i].strip()
                prop_match = self.property_pattern.match(line)
                if prop_match:
                    prop_name = prop_match.group(1)
                    prop_value = self._parse_value(prop_match.group(2))
                    properties[prop_name] = prop_value
                i += 1
        else:
            i = start_idx + 1
        
        # Add the object to the program
        obj = Object(name=name, properties=properties)
        self.program.add_object(obj)
        
        return i
    
    def _parse_atom(self, name: str, lines: List[str], start_idx: int) -> int:
        """Parse an atom block and return the new line index"""
        properties = {}
        
        # Get indentation level for the block
        if start_idx + 1 < len(lines):
            block_indent = self._get_indent_level(lines[start_idx + 1])
            
            # Process properties
            i = start_idx + 1
            nested_props = {}
            current_nested = None
            nested_indent = None
            
            while i < len(lines):
                # Check if we're still within the block
                indent = self._get_indent_level(lines[i])
                if indent < block_indent:
                    break
                    
                line = lines[i].strip()
                
                if indent > block_indent and current_nested:
                    # This is a nested property
                    if nested_indent is None:
                        nested_indent = indent
                    
                    if indent >= nested_indent:
                        prop_match = self.property_pattern.match(line)
                        if prop_match:
                            nested_key = prop_match.group(1)
                            nested_value = self._parse_value(prop_match.group(2))
                            nested_props[nested_key] = nested_value
                else:
                    # This is a top-level property
                    prop_match = self.property_pattern.match(line)
                    if prop_match:
                        prop_name = prop_match.group(1)
                        
                        # Check if this is the start of a nested block
                        if prop_name and ":" in line and not self._is_simple_value(prop_match.group(2)):
                            # Start a new nested property
                            if current_nested:
                                properties[current_nested] = nested_props
                            current_nested = prop_name
                            nested_props = {}
                            nested_indent = None
                        else:
                            # Regular property
                            prop_value = self._parse_value(prop_match.group(2))
                            properties[prop_name] = prop_value
                
                i += 1
            
            # Add the last nested property if any
            if current_nested:
                properties[current_nested] = nested_props
        else:
            i = start_idx + 1
        
        # Add the atom to the program
        atom = Atom(name=name, properties=properties)
        self.program.add_atom(atom)
        
        return i
    
    def _parse_field(self, name: str, lines: List[str], start_idx: int) -> int:
        """Parse a field block and return the new line index"""
        properties = {}
        
        # Get indentation level for the block
        if start_idx + 1 < len(lines):
            block_indent = self._get_indent_level(lines[start_idx + 1])
            
            # Process properties
            i = start_idx + 1
            while i < len(lines) and self._get_indent_level(lines[i]) >= block_indent:
                line = lines[i].strip()
                prop_match = self.property_pattern.match(line)
                if prop_match:
                    prop_name = prop_match.group(1)
                    prop_value = self._parse_value(prop_match.group(2))
                    properties[prop_name] = prop_value
                i += 1
        else:
            i = start_idx + 1
        
        # Add the field to the program
        field = Field(name=name, properties=properties)
        self.program.add_field(field)
        
        return i
    
    def _parse_interaction(self, source: str, target: str, lines: List[str], start_idx: int) -> int:
        """Parse an interaction block and return the new line index"""
        properties = {}
        
        # Get indentation level for the block
        if start_idx + 1 < len(lines):
            block_indent = self._get_indent_level(lines[start_idx + 1])
            
            # Process properties
            i = start_idx + 1
            while i < len(lines) and self._get_indent_level(lines[i]) >= block_indent:
                line = lines[i].strip()
                prop_match = self.property_pattern.match(line)
                if prop_match:
                    prop_name = prop_match.group(1)
                    prop_value = self._parse_value(prop_match.group(2))
                    properties[prop_name] = prop_value
                i += 1
        else:
            i = start_idx + 1
        
        # Add the interaction to the program
        interaction = Interaction(source=source, target=target, properties=properties)
        self.program.add_interaction(interaction)
        
        return i
    
    def _get_indent_level(self, line: str) -> int:
        """Get the indentation level of a line"""
        return len(line) - len(line.lstrip())
    
    def _parse_value(self, value_str: str) -> Any:
        """Parse a property value"""
        value_str = value_str.strip()
        
        # Check for array
        if value_str.startswith('[') and value_str.endswith(']'):
            # Simple array parsing - this could be improved
            inner = value_str[1:-1].strip()
            if not inner:
                return []
                
            # Split by commas, respecting nesting
            items = []
            current = ""
            brace_level = 0
            for char in inner + ',':  # Add comma to process the last item
                if char == ',' and brace_level == 0:
                    items.append(self._parse_value(current))
                    current = ""
                else:
                    if char == '[':
                        brace_level += 1
                    elif char == ']':
                        brace_level -= 1
                    current += char
            
            return items
        
        # Check for function call
        if '(' in value_str and value_str.endswith(')'):
            func_name, args_str = value_str.split('(', 1)
            func_name = func_name.strip()
            args_str = args_str[:-1].strip()  # Remove the closing paren
            
            # Parse arguments
            args = []
            kwargs = {}
            
            if args_str:
                # Split by commas, respecting nesting
                arg_parts = []
                current = ""
                brace_level = 0
                for char in args_str + ',':  # Add comma to process the last item
                    if char == ',' and brace_level == 0:
                        arg_parts.append(current.strip())
                        current = ""
                    else:
                        if char == '[':
                            brace_level += 1
                        elif char == ']':
                            brace_level -= 1
                        current += char
                
                for arg in arg_parts:
                    if '=' in arg:
                        key, val = arg.split('=', 1)
                        kwargs[key.strip()] = self._parse_value(val.strip())
                    else:
                        args.append(self._parse_value(arg))
            
            return {
                "function": func_name,
                "args": args,
                "kwargs": kwargs
            }
        
        # Check for boolean
        if value_str.lower() == 'true':
            return True
        if value_str.lower() == 'false':
            return False
        
        # Check for range (0..10)
        if '..' in value_str and all(part.strip().isdigit() for part in value_str.split('..', 1)):
            return value_str
        
        # Check for number
        try:
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            # If all else fails, return as string
            return value_str
    
    def _is_simple_value(self, value_str: str) -> bool:
        """Check if a value string represents a simple value or a block"""
        value_str = value_str.strip()
        return bool(value_str) and value_str[0] not in ('[', '{') 