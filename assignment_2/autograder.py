import os
import nbformat
from nbconvert import PythonExporter
import importlib.util
import sys
import math
import pandas as pd
from sklearn.datasets import load_iris as load_dataset
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tqdm import tqdm
from contextlib import redirect_stdout
import io

# -----------------------------
# Configuration
# -----------------------------
sys.dont_write_bytecode = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_FOLDER = os.path.join(BASE_DIR, "notebooks")
OUTPUT_CSV = os.path.join(BASE_DIR, "a2_grades.csv")
GRADE_DISTRIBUTION = {
    "1.1": 1,
    "1.2": 3,
    "2.1": 1,
    "2.2": 1,
    "2.3": 1,
    "2.4": 3,
    "2.5": 2,
    # "2.6": 2,
    # "2.7": 1,
    "3": 2
}

# -----------------------------
# Loading Functions
# -----------------------------
def notebook_to_module(notebook_path, module_path):
    """Convert a notebook to a Python script"""
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    exporter = PythonExporter()
    source, _ = exporter.from_notebook_node(nb)
    with open(module_path, "w") as f:
        f.write(source)

def import_module_from_path(module_name, module_path):
    """Dynamically import a module from a .py file"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    f = io.StringIO()
    with redirect_stdout(f):   # capture all prints
        spec.loader.exec_module(module)
    return module
    
# -----------------------------
# Task 1.1
# -----------------------------
TC_1_1 = [
    # (y_left, y_right)
    # General test case
    ([3, 4, 5], [8, 9]),
    ([1, 1, 1], [1, 1]),
    ([10], [20]),
    ([2, 2, 2, 2], [5]),
    ([1, 5], [3, 7]),
    ([0, 0, 0], [0]),
    ([42], [99]),
    ([1, 2, 3, 4, 5], [100]),
    ([7, 7, 7], [7, 7]),

    # Negative Values
    ([-1, -2, -3], [-4, -5]),
    ([-10, 0, 10], [5, -5]),
    ([1000, -1000], [0]),
    
    # Float Values
    ([1.1, 2.2, 3.3], [4.4, 5.5]),
    
    # Empty Lists. Need to mention that this is possible.
    # ([], [2, 4, 6]),
    # ([], [4, 4, 4]),
    # ([7, 7], []),
    # ([], []),
]

def calculate_regionrss_solution(y_left, y_right):
    def rss(y):
        if len(y) == 0:
            return 0.0
        mean_y = sum(y) / len(y)
        squared_differences = [(value - mean_y) ** 2 for value in y]
        return sum(squared_differences)

    total_rss = rss(y_left) + rss(y_right)
    return total_rss

# -----------------------------
# Task 1.2
# -----------------------------
TC_1_2 = [
    # (X, y, X_test, max_depth)
    # General test case
    ([1, 2, 3, 4, 5], [2, 4, 6, 8, 10], [1.5, 3.5, 5], 2),
    ([0, 1, 2, 3], [5, 7, 9, 11], [0.5, 2.5], 2), 
    ([1, 2, 3, 4, 5], [10, 10, 20, 20, 20], [1.5, 3.5, 5.5], 2),
    ([1, 2, 3], [5, 5, 5], [1.5, 2.5], 2),
    ([1, 2], [3, 6],[1.5], 2),
    ([3, 1, 4, 2], [30, 10, 40, 20], [1.5, 3.5], 2),
    ([42], [100], [0, 42, 100], 5),
    ([0, 0, 0], [0, 0, 0], [0, 0.1, -0.1], 2),
    ([1, 2, 2, 2, 3, 4], [10, 5, 15, 5, 20, 25], [2, 2.1, 3], 2),
    ([1, 2, 3], [10, 20, 30], [-999, 999], 2),
    ([1, 1, 1, 10], [5, 5, 5, 100], [1, 2, 5, 10], 2),

    # Depth of 0
    ([1, 2, 3], [3, 6, 9], [0, 2, 5], 0),

    # Depth very high -> Stop at leaf of 1 element
    ([1, 2, 3, 4], [10, 20, 30, 40], [1, 2, 3, 4], 100),

    # Empty Prediction Array
    # ([1, 2, 3], [2, 4, 6], [], 2),
]

class DecisionTreeRegressor_solution:
    def __init__(self, max_depth=2):
        self.max_depth = max_depth
        self.tree = None  

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
    def _build_tree(self, X, y, depth):
        unique_X = sorted(set(X))
        if depth == self.max_depth or len(unique_X) <= 1:
            return sum(y) / len(y) 
        
        best_rss = float('inf')
        best_split = None
        
        for i in range(1, len(unique_X)):
            if unique_X[i] == unique_X[i - 1]:
                continue  # Skip redundant splits
            split_value = (unique_X[i - 1] + unique_X[i]) / 2
            left_y = [y[j] for j in range(len(X)) if X[j] < split_value]
            right_y = [y[j] for j in range(len(X)) if X[j] >= split_value]
            
            current_rss = calculate_regionrss_solution(left_y, right_y)
            
            if current_rss < best_rss:
                best_rss = current_rss
                best_split = split_value
        
        if best_split is None:
            return sum(y) / len(y)  
        
        left_X = [X[i] for i in range(len(X)) if X[i] < best_split]
        left_y = [y[i] for i in range(len(X)) if X[i] < best_split]
        right_X = [X[i] for i in range(len(X)) if X[i] >= best_split]
        right_y = [y[i] for i in range(len(X)) if X[i] >= best_split]
        
        return {
            'split_value': best_split,
            'left': self._build_tree(left_X, left_y, depth + 1),
            'right': self._build_tree(right_X, right_y, depth + 1)
        }

    def predict_one(self, x, node=None):
        if node is None:
            node = self.tree

        if not isinstance(node, dict):
            return node 
        
        if x < node['split_value']:
            return self.predict_one(x, node['left'])
        else:
            return self.predict_one(x, node['right'])
    
    def predict(self, X):
        return [self.predict_one(x) for x in X]

# -----------------------------
# Task 2.1
# -----------------------------
TC_2_1 = [
    # (y, n_unique_classes)
    # General test case
    (['yes', 'no', 'yes', 'yes', 'no'], 2),
    (['yes', 'yes', 'yes','no','maybe','maybe','no','maybe'], 3),
    (['cat', 'dog', 'cat', 'fish', 'dog', 'cat'], 3),
    ([1, 1, 0, 0, 1, 1], 2),
    (['A', 'B', 'C', 'A', 'B', 'C', 'A'], 3),
    (['a', 'b', 'a', 'b'], 2),
    (['x', 'y', 'z', 'w'], 4),
    ([True, False, True, False], 4),
    (['1', '2', '3', '2', '1'], 2),
    ([0, 1, 2, 1, 0], 4),

    # Single Class, handle log of base 1
    (['yes', 'yes', 'yes'], 1),

    # Empty List
    # ([], 2),
]

def compute_entropy_solution(y, n_unique_classes=2):
    if n_unique_classes == 1:
        return 0
    label_counts = {}
    for label in y:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    total = len(y)
    entropy = 0.0
    for count in label_counts.values():
        p = count / total
        entropy -= p * math.log(p, n_unique_classes)
    
    return entropy

# -----------------------------
# Task 2.2
# -----------------------------
TC_2_2 = [
    # (parent_y, list_of_child_ys, n_unique_classes)
    # General test case
    (['yes', 'no', 'yes', 'no'], [['yes', 'yes'], ['no', 'no']], 2),
    (['yes', 'no', 'yes', 'no'], [['yes', 'no'], ['yes', 'no']], 2),
    (['yes', 'no', 'yes', 'no', 'yes', 'no'], [['yes', 'yes'], ['no', 'no'], ['yes', 'no']], 2),
    (['p', 'p', 'n', 'p', 'n'], [['p', 'p'], ['n', 'p', 'n']], 2),
    (['one', 'one', 'two', 'three', 'three'], [['one', 'one'], ['two', 'three', 'three']], 2),
    (['x', 'y', 'z'], [['x'], ['y', 'z']], 2),
    (['1', '2', '1', '2'], [['1', '1'], ['2', '2']], 2),
    ([True, False, True, False], [[True, True], [False, False]], 2),
    (['yes', 'yes', 'yes'], [['yes'], ['yes', 'yes']], 2),
    (['a', 'b', 'c', 'a', 'b'], [['a', 'b'], ['c', 'a', 'b']], 3),
    (['yes', '1', 'no', '2'], [['yes', '1'], ['no', '2']], 4),
    ([0, 1, 2, 1, 0], [[0, 1], [2, 1, 0]], 3),
    (['dog', 1, False], [['dog', 1], [False]], 3),

    # No Split
    (['yes', 'no', 'yes'], [['yes', 'no', 'yes']], 2),

    # Empty Parent. Need to mention that this is possible.
    # ([], [[]], 2),
]

def information_gain_solution(parent_y, list_of_child_ys, n_unique_classes=2):
    n = len(parent_y)
    if n == 0:
        return 0.0 # No gain if parent node is empty

    entropy_parent = compute_entropy_solution(parent_y, n_unique_classes)
    weighted_entropy = 0.0

    for child_y in list_of_child_ys:
        weight = len(child_y) / n
        if len(child_y) > 0:
            weighted_entropy += weight * compute_entropy_solution(child_y, n_unique_classes)
    
    gain = entropy_parent - weighted_entropy
    return gain

# -----------------------------
# Task 2.3
# -----------------------------
TC_2_3 = [
    # (y,)
    # General test case
    (['yes', 'no', 'yes', 'no', 'yes'],),
    (['yes', 'no', 'yes', 'no'],),
    ([1, 2, 2, 3, 2],),
    (['cat'],),
    ([],),
    (['x', 'x', 'x'],),
    ([3, 1, 3, 2, 2, 1, 1],),
    ([True, False, True],),
    (['1', '2', '1', '3'],),

    # Tie-breaker
    (['b', 'a'],),
    (['x', 'z'],),
    (['red', 'green', 'blue'],),
]

def majority_class_solution(y):
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1

        max_count = -1 
        majority_class = None
        
        # Get labels in sorted order to ensure deterministic tie-breaking (smallest label)
        sorted_labels = sorted(counts.keys())

        for label in sorted_labels:
            count = counts[label]
            if count > max_count:
                max_count = count
                majority_class = label
        
        return majority_class

# -----------------------------
# Task 2.4
# -----------------------------
TC_2_4 = [
    # (X, y, n_unique_classes)
    # General test case
    ([['red'], ['blue'], ['red'], ['green'], ['blue']], ['yes', 'no', 'yes', 'no', 'no'], 2),
    ([[2.0], [4.0], [6.0], [8.0], [10.0]], ['yes', 'yes', 'no', 'no', 'no'], 2),
    ([['same'], ['same'], ['same']], ['yes', 'yes', 'yes'], 1),
    ([['x'], ['y'], ['x'], ['z']], [2, 1, 2, 1], 2),
    ([[True], [False], [True], [False]], ['yes', 'no', 'yes', 'no'], 2),
    ([['1'], ['2'], ['3'], ['1']], ['a', 'b', 'c', 'a'], 3),

    # Tie-breaker
    ([['a', 2.0], ['a', 4.0], ['b', 6.0], ['b', 8.0]], ['yes', 'yes', 'no', 'no'], 2),

    # No valid split
    ([[0], [0], [0]], ['yes', 'no', 'yes'], 2),    
    ([[1], [2], [3], [4]], ['yes', 'yes', 'yes', 'yes'], 1),
    ([[8.0], [2.0], [10.0], [4.0], [6.0]], ['no', 'yes', 'no', 'yes', 'no'], 2),

    # Empty dataset
    ([], [], 2),

    # Complex tie-breaker
    (
        [
            [2.5, 'A',  1.2, 10.0, 'red'],
            [3.5, 'B',  3.1, 15.0, 'blue'],
            [2.0, 'A',  1.3,  9.5, 'red'],
            [4.0, 'B',  3.0, 14.0, 'green'],
            [3.8, 'A',  2.9, 13.5, 'blue'],
            [4.1, 'B',  3.2, 14.5, 'green'],
            [1.8, 'A',  1.1, 10.5, 'red'],
            [3.6, 'B',  3.3, 15.5, 'blue'],
        ],
        ['yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no'],
        2
    ),
    (
        [
            ['red', 2.5, 'A',  1.2, 10.0],
            ['blue', 3.5, 'B',  3.1, 15.0],
            ['red', 2.0, 'A',  1.3,  9.5],
            ['green', 4.0, 'B',  3.0, 14.0],
            ['blue', 3.8, 'A',  2.9, 13.5],
            ['green', 4.1, 'B',  3.2, 14.5],
            ['red', 1.8, 'A',  1.1, 10.5],
            ['blue', 3.6, 'B',  3.3, 15.5],
        ],
        ['yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no'],
        2
    ),
    (
        [
            ['A', 1.2, 10.0, 'red', 2.5],
            ['B', 3.1, 15.0, 'blue', 3.5],
            ['A', 1.3, 9.5, 'red', 2.0],
            ['B', 3.0, 14.0, 'green', 4.0],
            ['A', 2.9, 13.5, 'blue', 3.8],
            ['B', 3.2, 14.5, 'green', 4.1],
            ['A', 1.1, 10.5, 'red', 1.8],
            ['B', 3.3, 15.5, 'blue', 3.6],
        ],
        ['yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no'],
        2
    )
]

def find_best_split_solution(X, y, n_unique_classes=2):
    best_gain = -1.0
    best_feature_idx = None
    best_split_type = None
    best_split_details = None
    featureType = None
    
    if len(X) == 0:
        return (best_gain, best_feature_idx, best_split_type, best_split_details)
    
    for featureInd in range(len(X[0])): # each feature in input
        try:
            X[0][featureInd] / 2
            featureType = "continuous"
        except:
            featureType = "categorical"

        if featureType == "continuous":
            pairs = sorted(zip(X, y), key=lambda pair: pair[0][featureInd])
            X_sorted, y_sorted = map(list, zip(*pairs))
            for ind in range(1, len(X_sorted)): # Go through all splits
                if X_sorted[ind - 1] == X_sorted[ind]:
                    continue
                gain = information_gain_solution(y, [y_sorted[:ind], y_sorted[ind:]], n_unique_classes)
                if gain > best_gain and gain > 0:
                    best_gain = gain
                    best_feature_idx = featureInd
                    best_split_type = featureType
                    best_split_details = {
                        "split_value" : (X_sorted[ind - 1][featureInd] + X_sorted[ind][featureInd]) / 2,
                        "left_X" : X_sorted[:ind],
                        "left_y" : y_sorted[:ind],
                        "right_X" : X_sorted[ind:],
                        "right_y" : y_sorted[ind:]
                    }
                    
        else: # categorical feature
            childArraysDict = {}
            for ind in range(len(X)):
                if X[ind][featureInd] not in childArraysDict:
                    childArraysDict[X[ind][featureInd]] = {'X': [], 'y': []}
                childArraysDict[X[ind][featureInd]]['y'].append(y[ind])
                childArraysDict[X[ind][featureInd]]['X'].append(X[ind])
            childArrays = [x['y'] for x in childArraysDict.values()]
            if len(childArrays) == 1:
                continue
            gain = information_gain_solution(y, childArrays, n_unique_classes)
            if gain > best_gain and gain > 0:
                best_gain = gain
                best_feature_idx = featureInd
                best_split_type = featureType
                best_split_details = childArraysDict

    return (best_gain, best_feature_idx, best_split_type, best_split_details)

# -----------------------------
# Task 2.5
# -----------------------------
TC_2_5 = [
    # (X, y, depth, max_depth, n_unique_classes)
    # General test case
    ([[1], [2], [3]], [0, 1, 0], 2, 2, 2),
    ([[1], [2], [3]], [0, 0, 0], 0, 5, 2),
    ([[1], [2], [10], [12]], [0, 0, 1, 1], 0, 2, 2),
    ([['Red'], ['Blue'], ['Red'], ['Green']], [0, 1, 0, 2], 0, 2, 3),
    ([[1], [2], [3]], [0, 0, 0], 0, 5, 2),
    ([[10], [2], [12], [3], [1]], [0, 1, 0, 1, 1], 0, 2, 2),
    ([['A'], ['B'], ['A'], ['C']], [0, 1, 0, 1], 0, 2, 2),
    ([[10], [2], [12], [3], [1]], [0, 1, 0, 1, 1], 0, 1, 2),
    ([[1], [2], [3]], [0, 1, 0], 0, 0, 2),
    ([[1], [2], [3], [4]], [0, 1, 2, 3], 0, 5, 4),
    ([[5], [5], [5]], [1, 1, 0], 0, 3, 2),
    ([[10, 'A'], [20, 'B'], [15, 'A'], [25, 'C']], [0, 1, 0, 1], 0, 3, 2),
    ([[42]], [1], 0, 3, 1),

    # Empty List
    ([], [0, 1, 0], 0, 5, 2),
    ([], [], 0, 3, 0),

    # Complex Case
    (
        [
            [1.0, 'A', 0, 'X', 0.1, 10.0],
            [1.2, 'A', 1, 'X', 0.2, 10.5],
            [1.4, 'A', 0, 'X', 0.3, 11.0],
            [2.0, 'B', 1, 'Y', 0.4, 11.5],
            [2.2, 'B', 0, 'Y', 0.5, 12.0],
            [2.4, 'B', 1, 'Y', 0.6, 12.5],
            [3.0, 'C', 0, 'Z', 0.7, 13.0],
            [3.2, 'C', 1, 'Z', 0.8, 13.5],
            [3.4, 'C', 0, 'Z', 0.9, 14.0],
            [4.0, 'D', 1, 'X', 1.0, 14.5],
            [4.2, 'D', 0, 'Y', 1.1, 15.0],
            [4.4, 'D', 1, 'Z', 1.2, 15.5],
            [5.0, 'E', 0, 'X', 1.3, 16.0],
            [5.2, 'E', 1, 'Y', 1.4, 16.5],
            [5.4, 'E', 0, 'Z', 1.5, 17.0],
        ],
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        0,
        4,
        5
    ),
]

def build_tree_recursive_solution(X, y, depth, max_depth, n_unique_classes):
    # Base Cases:
    if depth == max_depth or len(set(y)) == 1:
        return majority_class_solution(y)
    
    if not X:
        return majority_class_solution(y) 

    # Pass n_unique_classes to find_best_split
    best_overall_gain, best_overall_feature_index, best_overall_split_type, best_overall_split_details = \
        find_best_split_solution(X, y, n_unique_classes)
    
    if best_overall_gain <= 0: 
        return majority_class_solution(y) 

    current_nodemajority_class = majority_class_solution(y)

    # Build child nodes based on the best overall split found
    if best_overall_split_type == 'continuous':
        return {
            'feature': best_overall_feature_index,
            'split_type': 'continuous',
            'split_value': best_overall_split_details['split_value'],
            # Pass n_unique_classes recursively
            'left': build_tree_recursive_solution(best_overall_split_details['left_X'], best_overall_split_details['left_y'], depth + 1, max_depth, n_unique_classes),
            'right': build_tree_recursive_solution(best_overall_split_details['right_X'], best_overall_split_details['right_y'], depth + 1, max_depth, n_unique_classes),
            'default_prediction': current_nodemajority_class 
        }
    else: # best_overall_split_type == 'categorical'
        children_nodes = {}
        for category_val, data in best_overall_split_details.items():
            # Pass n_unique_classes recursively
            children_nodes[category_val] = build_tree_recursive_solution(data['X'], data['y'], depth + 1, max_depth, n_unique_classes)
        
        return {
            'feature': best_overall_feature_index,
            'split_type': 'categorical',
            'children_map': children_nodes, 
            'default_prediction': current_nodemajority_class 
        }
    
# -----------------------------
# Task 2.6
# -----------------------------
TC_2_6 = [

]

def predict_one_instance_solution(x_instance, tree_node):
    if not isinstance(tree_node, dict): # Leaf node (contains the predicted class directly)
        return tree_node
    
    # Check if the feature index exists in the input 'x_instance'
    feature_idx = tree_node['feature']
    if feature_idx >= len(x_instance):
        # This indicates an input 'x_instance' is shorter than expected by the tree structure.
        # Fallback to the default prediction stored at the node.
        return tree_node.get('default_prediction', None) 

    feature_val = x_instance[feature_idx]

    if tree_node['split_type'] == 'continuous':
        split_value = tree_node['split_value']
        if feature_val < split_value:
            return predict_one_instance_solution(x_instance, tree_node['left'])
        else:
            return predict_one_instance_solution(x_instance, tree_node['right'])
    
    else: # tree_node['split_type'] == 'categorical':
        # Use .get() with the default_prediction as fallback for unseen categories
        next_node = tree_node['children_map'].get(feature_val)
        if next_node is not None:
             return predict_one_instance_solution(x_instance, next_node)
        else:
             # Unseen category encountered, use the default prediction for this node
             return tree_node['default_prediction']
        
# -----------------------------
# Task 2.7
# -----------------------------
TC_2_7 = [

]

class DecisionTreeClassifier_Solution:
    def __init__(self, max_depth=2):
        self.max_depth = max_depth
        self.tree = None 
        self.n_unique_classes = None 

    def fit(self, X, y):
        self.n_unique_classes = len(set(y))
        if self.n_unique_classes <= 1:
            self.tree = majority_class_solution(y) 
            return
        self.tree = build_tree_recursive_solution(X, y, depth=0, max_depth=self.max_depth, n_unique_classes=self.n_unique_classes)

    def predict(self, X):
        if not isinstance(self.tree, dict):
            return [self.tree for _ in X]
        return [predict_one_instance_solution(x_instance, self.tree) for x_instance in X]
    
# -----------------------------
# Task 3
# -----------------------------
ACCURACY_THRESHOLD = 0.5

# -----------------------------
# Generic Grader Functions
# -----------------------------
def default_assert(output, expected):
    return output == expected

def float_assert(output, expected, tolerance=1e-5):
    return abs(output - expected) < tolerance

def list_float_assert(output, expected, tolerance=1e-5):
    return all(abs(o - e) < tolerance for o, e in zip(output, expected))

def q24_assert(output, expected, tolerance=1e-5):
    output_gain, output_feature_idx, output_split_type, output_split_details = output
    expected_gain, expected_feature_idx, expected_split_type, expected_split_details = expected

    gain_assert = float_assert(output_gain, expected_gain, tolerance)
    feature_idx_assert = default_assert(output_feature_idx, expected_feature_idx)
    split_type_assert = default_assert(output_split_type, expected_split_type)

    if not (gain_assert and feature_idx_assert and split_type_assert):
        return False
    
    split_details_assert = True
    if expected_split_type == "categorical":
        split_details_assert = split_details_assert and default_assert(output_split_details, expected_split_details)
    elif expected_split_type == "continuous":
        split_details_assert = split_details_assert and float_assert(output_split_details['split_value'], expected_split_details['split_value'], tolerance)
        split_details_assert = split_details_assert and default_assert(output_split_details['left_X'].sort(), expected_split_details['left_X'].sort())
        split_details_assert = split_details_assert and default_assert(output_split_details['left_y'].sort(), expected_split_details['left_y'].sort())
        split_details_assert = split_details_assert and default_assert(output_split_details['right_X'].sort(), expected_split_details['right_X'].sort())
        split_details_assert = split_details_assert and default_assert(output_split_details['right_y'].sort(), expected_split_details['right_y'].sort())

    return split_details_assert

def grade_function(student_fn, solution_fn, test_cases, check_fn):
    try:
        for args in test_cases:
            output = student_fn(*args)
            expected = solution_fn(*args)
            if not check_fn(output, expected):
                return 0
        return 1
    except Exception:
        return 0
    
def grade_class(student_class, solution_class, test_cases, check_fn):
    try:
        for X_train, y_train, X_test, k in test_cases:
            output_class = student_class(k)
            output_class.fit(X_train, y_train)
            output = output_class.predict(X_test)

            expected_class = solution_class(k)
            expected_class.fit(X_train, y_train)
            expected = expected_class.predict(X_test)
            if not check_fn(output, expected):
                return 0
        return 1
    except Exception:
        return 0
    
def grade_model(student_fn):
    dataset = load_dataset()
    X, y = dataset.data, dataset.target

    # Split into training and test sets
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, test_idx in cv.split(X, y):
            X_train_cv, X_test_cv = X[train_idx], X[test_idx]
            y_train_cv, y_test_cv = y[train_idx], y[test_idx]
            model_cv = student_fn(X_train_cv, y_train_cv)
            scorer = get_scorer("accuracy")
            score = scorer(model_cv, X_test_cv, y_test_cv)
            scores.append(score)

        scores = np.array(scores)
        accuracy_score = scores.mean()
        return 1 if accuracy_score > ACCURACY_THRESHOLD else 0
    except Exception:
        return 0
    
# -----------------------------
# Autograde Workflow
# -----------------------------
TASKS = [
    # (type, name, solution, check_type, test_cases, grade_weight)
    ("function", "calculate_regionrss", calculate_regionrss_solution, float_assert, TC_1_1, GRADE_DISTRIBUTION["1.1"]),
    ("class", "DecisionTreeRegressor", DecisionTreeRegressor_solution, list_float_assert, TC_1_2, GRADE_DISTRIBUTION["1.2"]),
    ("function", "compute_entropy", compute_entropy_solution, float_assert, TC_2_1, GRADE_DISTRIBUTION["2.1"]), 
    ("function", "information_gain", information_gain_solution, float_assert, TC_2_2, GRADE_DISTRIBUTION["2.2"]), 
    ("function", "majority_class", majority_class_solution, default_assert, TC_2_3, GRADE_DISTRIBUTION["2.3"]), 
    ("function", "find_best_split", find_best_split_solution, q24_assert, TC_2_4, GRADE_DISTRIBUTION["2.4"]),
    ("function", "build_tree_recursive", build_tree_recursive_solution, default_assert, TC_2_5, GRADE_DISTRIBUTION["2.5"]),
    # ("function", "predict_one_instance", predict_one_instance_solution, ?, TC_2_6, GRADE_DISTRIBUTION["2.6"]),
    # ("class", "DecisionTreeClassifier", DecisionTreeClassifier_Solution, ?, TC_2_7, GRADE_DISTRIBUTION["2.7"]),
    ("model", "train_model", None, None, None, GRADE_DISTRIBUTION["3"])
]

def autograde_folder(folder):
    rows = []
    fails = []
    for filename in tqdm(os.listdir(folder)):
        if not filename.endswith(".ipynb"):
            continue

        nb_path = os.path.join(folder, filename)
        module_path = nb_path.replace(".ipynb", ".py")
        student_number = filename[:-6]
        
        # Convert notebook -> module
        notebook_to_module(nb_path, module_path)
        
        # Import module
        module_name = filename.replace(".ipynb", "")
        try:
            module = import_module_from_path(module_name, module_path)
        except Exception:
            # Catch notebooks that fail to run
            fails.append(student_number)
            os.remove(module_path)
            continue

        # Grade function
        total_score = 0

        row = [student_number]
        for type, task, solution_fn, check_fn, test_cases, weight in TASKS:
            student_fn = getattr(module, task)
            if type == "function":
                score = grade_function(student_fn, solution_fn, test_cases, check_fn) * weight
            elif type == "class":
                score = grade_class(student_fn, solution_fn, test_cases, check_fn) * weight
            # elif type == "q5":
            #     score = grade_q5(student_fn) * weight

            row.append(score)
            total_score += score
        row.append(total_score)
        rows.append(row)

        os.remove(module_path)

    print(f"Failed to compile: {fails}")
    columns = ["student_number"] + sorted(GRADE_DISTRIBUTION.keys()) + ["total"]
    return pd.DataFrame(rows, columns=columns)

# -----------------------------
# Save report to CSV
# -----------------------------
def save_to_csv(report, csv_file):
    report.to_csv(csv_file, index = False)

# -----------------------------
# Run autograder
# -----------------------------
if __name__ == "__main__":
    final_report = autograde_folder(NOTEBOOK_FOLDER)
    save_to_csv(final_report, OUTPUT_CSV)
    print(f"Grading completed! Results saved to {OUTPUT_CSV}")
