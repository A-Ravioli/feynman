import sys

# Ensure the main function from the main module is executed
# when running the package with python -m feynman
if __package__ is None and not hasattr(sys, 'frozen'):
    # direct execution \"python feynman/__main__.py\"
    import os.path
    path = os.path.realpath(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(path)))

from .main import main

if __name__ == '__main__':
    main() 