import os
from pathlib import Path
from lark import Lark, Transformer, v_args, Token
from lark.indenter import Indenter

class PhysicaIndenter(Indenter):
    NL_type = '_NL'
    OPEN_PAREN_types = []
    CLOSE_PAREN_types = []
    INDENT_type = '_INDENT'
    DEDENT_type = '_DEDENT'
    tab_len = 4
    
    def process(self, stream):
        # Process the token stream to handle indentation
        for token in stream:
            # Skip comments and whitespace-only tokens
            if token.type == 'COMMENT' or (token.type == '_NL' and not token.value.strip()):
                yield token
                continue
                
            # Process normal tokens
            yield from super().process([token])

class PhysicaTransformer(Transformer):
    @v_args(inline=True)
    def string(self, s):
        return s[1:-1]
    
    @v_args(inline=True)
    def number(self, n):
        return float(n)
    
    @v_args(inline=True)
    def fraction(self, numerator, denominator):
        return float(numerator) / float(denominator)
    
    @v_args(inline=True)
    def boolean(self, b):
        return b.lower() == "true"
    
    def array(self, items):
        return list(items)
    
    def function_call(self, items):
        name = items[0]
        args = items[1:] if len(items) > 1 else []
        return {"function": name, "args": args}

class PhysicaParser:
    def __init__(self):
        grammar_path = Path(os.path.dirname(__file__)) / "grammar.lark"
        with open(grammar_path, "r") as f:
            grammar = f.read()
        
        # Create indenter and parser
        self.indenter = PhysicaIndenter()
        self.parser = Lark(grammar, parser="lalr", postlex=self.indenter)
        self.transformer = PhysicaTransformer()
    
    def parse(self, code):
        """Parse PhysicaLang code and return AST"""
        # Normalize line endings and ensure trailing newline
        code = code.replace('\r\n', '\n').replace('\r', '\n')
        if not code.endswith('\n'):
            code += '\n'
        
        # Process leading/trailing whitespace
        code = '\n'.join(line.rstrip() for line in code.split('\n'))
        
        try:
            tree = self.parser.parse(code)
            return tree
        except Exception as e:
            # Print more detailed error information
            print(f"Error parsing code:\n{code}")
            print(f"Error details: {str(e)}")
            raise
    
    def parse_and_transform(self, code):
        """Parse PhysicaLang code and transform to Python structures"""
        tree = self.parse(code)
        return self.transformer.transform(tree)

if __name__ == "__main__":
    # Basic test
    test_code = """
model test_model:
    type: classical
    time: 0..10
    resolution: 0.01

object ball:
    mass: 1.0
    position: [0, 0, 0]
    velocity: [1, 0, 0]

simulate test_model
"""
    parser = PhysicaParser()
    tree = parser.parse(test_code)
    print(tree.pretty()) 