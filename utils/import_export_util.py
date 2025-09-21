import ast
import importlib.util
import io
import math
import os
import re
import sys
from collections import Counter
from contextlib import redirect_stdout

import nbformat
import pandas as pd
from nbconvert import PythonExporter
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# -----------------------------
# Loading Functions
# -----------------------------
def notebook_to_module(notebook_path, module_path):
    """Convert a notebook to a Python script"""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    exporter = PythonExporter()
    source, _ = exporter.from_notebook_node(nb)
    with open(module_path, "w", encoding="utf-8") as f:
        f.write(source)

class KeepImportsAndDefs(ast.NodeTransformer):
    def visit_Module(self, node):
        # Keep imports and function/class definitions
        new_body = [
            n for n in node.body
            if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        node.body = new_body
        return node
    
def import_module_safe(module_name, module_path):
    # Read source
    with open(module_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Parse AST and keep only imports + functions/classes
    tree = ast.parse(source, filename=module_path)
    tree = KeepImportsAndDefs().visit(tree)
    ast.fix_missing_locations(tree)

    # Compile and execute in a new module
    code = compile(tree, module_path, "exec")
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(module_name, loader=None)
    )
    sys.modules[module_name] = module
    module.__dict__["print"] = lambda *args, **kwargs: None

    exec(code, module.__dict__)

    return module

# -----------------------------
# Export Functions
# -----------------------------
def save_to_csv(report, csv_file):
    report.to_csv(csv_file, index = False)

def save_fails(fails, txt_file):
    with open(txt_file, "w") as f:
        for item in fails:
            f.write(f"{item}\n")