import os
import sys

#Make Code/ importable (evaluation.py, pass_model.py) when pytest runs from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
