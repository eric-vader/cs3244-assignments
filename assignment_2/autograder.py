import math
import os
import re
import sys

import pandas as pd
from sklearn.datasets import load_iris
from tqdm import tqdm

from utils.grading_util import *
from utils.import_export_util import *

# -----------------------------
# Configuration
# -----------------------------
sys.dont_write_bytecode = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_FOLDER = os.path.join(BASE_DIR, "notebooks")
SCORE_CSV = os.path.join(BASE_DIR, "a2_grades.csv")
FAILS_TXT = os.path.join(BASE_DIR, "a2_fails.txt")
GRADE_DISTRIBUTION = {
    "1.1": 1,
    "1.2": 3,
    "2.1": 1,
    "2.2": 1,
    "2.3": 1,
    "2.4": 3,
    "2.5": 2,
    "2.6": 2,
    "2.7": 1,
    "3": 2
}

# -----------------------------
# Task 1.1
# -----------------------------
TC_1_1 = [
    # (y_left, y_right)
    {
        "point": 0.5,
        "desc": "General Case",
        "tc": [
            # General Case
            ([3, 4, 5], [8, 9]),
            ([1, 1, 1], [1, 1]),
            ([10], [20]),
            ([2, 2, 2, 2], [5]),
            ([1, 5], [3, 7]),
            ([0, 0, 0], [0]),
            ([42], [99]),
            ([1, 2, 3, 4, 5], [100]),
            ([7, 7, 7], [7, 7]),
            ([1.1, 2.2, 3.3], [4.4, 5.5]),
        ]
    }, {
        "point": 0.5,
        "desc": "Edge Case: Negative Input or Empty List (Only RSS for that split is 0, not for total)",
        "tc": [
            # Negative Input
            ([-1, -2, -3], [-4, -5]),
            ([-10, 0, 10], [5, -5]),
            ([1000, -1000], [0]),
            # Empty List (Only RSS for that split is 0, not for total). Need to mention that this is possible.
            ([], [2, 4, 6]),
            ([], [4, 4, 4]),
            ([7, 7], []),
            ([], []),
        ]
    },    
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
    {
        "point": 0.5,
        "desc": "General Case",
        "tc": [
            # General Case
            ([1, 2, 3, 4, 5], [2, 4, 6, 8, 10], [1.5, 3.5, 5], 2),
            ([0, 1, 2, 3], [5, 7, 9, 11], [0.5, 2.5], 2), 
            ([1, 2, 3, 4, 5], [10, 10, 20, 20, 20], [1.5, 3.5, 5.5], 2),
            ([1, 2, 3], [5, 5, 5], [1.5, 2.5], 2),
            ([3, 1, 4, 2], [30, 10, 40, 20], [1.5, 3.5], 2),
            ([1, 2, 3], [10, 20, 30], [-999, 999], 2),
            ([1, 1, 1, 10], [5, 5, 5, 100], [1, 2, 5, 10], 2),
            ([1, 2, 3, 4], [10, 20, 30, 40], [1, 2, 3, 4], 100),
        ]
    }, {
        "point": 0.25,
        "desc": "Edge Case: X_test = Continuous Split Value (Split is Midpoint, Go Right if Equal)",
        "tc": [
            # X_test = Continuous Split Value (Split is Midpoint, Go Right if Equal)
            ([1, 2], [3, 6], [1.5], 2),
            ([1, 2, 2, 2, 3, 4], [10, 5, 15, 5, 20, 25], [2, 2.1, 3], 2),
        ]
    }, {
        "point": 0.25,
        "desc": "Edge Case: Depth of 0, 1 Unique X, or Empty Prediction Array",
        "tc": [
            # Depth of 0
            ([1, 2, 3], [3, 6, 9], [0, 2, 5], 0),
            # 1 Unique X
            ([0, 0, 0], [0, 0, 0], [0, 0.1, -0.1], 2),
            ([42], [100], [0, 42, 100], 5),
            # Empty Prediction Array
            ([1, 2, 3], [2, 4, 6], [], 2),
        ]
    },    
]

class DTRegressor_solution:
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
    {
        "point": 0.5,
        "desc": "General Case (Use n_unique_class for Log Base)",
        "tc": [
            # General Case
            (['yes', 'no', 'yes', 'yes', 'no'], 2),
            (['yes', 'yes', 'yes','no','maybe','maybe','no','maybe'], 3),
            (['cat', 'dog', 'cat', 'fish', 'dog', 'cat'], 3),
            ([1, 1, 0, 0, 1, 1], 2),
            (['A', 'B', 'C', 'A', 'B', 'C', 'A'], 3),
            (['a', 'b', 'a', 'b'], 2),
            (['x', 'y', 'z', 'w'], 4),
            (['1', '2', '3', '2', '1'], 3),
        ]
    }, {
        "point": 0.5,
        "desc": "Edge Case: 1 Unique Class (0 Entropy) or n_unique class > Distinct Labels (Use n_unique_class for Log Base)",
        "tc": [
            # 1 Unique Class (0 Entropy)
            (['yes', 'yes', 'yes'], 1),
            # n_unique class > Distinct Labels (Use n_unique_class for Log Base)
            ([True, False, True, False], 4),
            ([0, 1, 2, 1, 0], 4),
        ]
    },    
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
    {
        "point": 0.5,
        "desc": "General Case (Use n_unique_class for All Log Bases)",
        "tc": [
            # General Case
            (['yes', 'no', 'yes', 'no'], [['yes', 'yes'], ['no', 'no']], 2),
            (['yes', 'no', 'yes', 'no'], [['yes', 'no'], ['yes', 'no']], 2),
            (['yes', 'no', 'yes', 'no', 'yes', 'no'], [['yes', 'yes'], ['no', 'no'], ['yes', 'no']], 2),
            (['p', 'p', 'n', 'p', 'n'], [['p', 'p'], ['n', 'p', 'n']], 2),
            (['one', 'one', 'two', 'three', 'three'], [['one', 'one'], ['two', 'three', 'three']], 3),
            (['x', 'y', 'z'], [['x'], ['y', 'z']], 3),
            (['1', '2', '1', '2'], [['1', '1'], ['2', '2']], 2),
            ([True, False, True, False], [[True, True], [False, False]], 2),
            (['a', 'b', 'c', 'a', 'b'], [['a', 'b'], ['c', 'a', 'b']], 3),
            (['yes', '1', 'no', '2'], [['yes', '1'], ['no', '2']], 4),
        ]
    }, {
        "point": 0.5,
        "desc": "Edge Case: Empty Parent, n_unique class > Distinct Labels (Use n_unique_class for Log Base), or No Valid Split",
        "tc": [
            # Empty Parent. Need to mention that this is possible.
            ([], [[]], 2),
            # n_unique class > Distinct Labels (Use n_unique_class for Log Base)
            (['yes', 'yes', 'yes'], [['yes'], ['yes', 'yes']], 2),
            ([0, 1, 2, 1, 0], [[0, 1], [2, 1, 0]], 4),
            # No Valid Split
            (['yes', 'no', 'yes'], [['yes', 'no', 'yes']], 2),
        ]
    },    
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
    {
        "point": 0.5,
        "desc": "General Case",
        "tc": [
            # General Case
            (['yes', 'no', 'yes', 'no', 'yes'],),
            ([1, 2, 2, 3, 2],),
            (['cat'],),
            (['x', 'x', 'x'],),
            ([3, 1, 3, 2, 2, 1, 1],),
            ([True, False, True],),
            (['1', '2', '1', '3'],),
        ]
    }, {
        "point": 0.5,
        "desc": "Edge Case: Tie-Breaker Check",
        "tc": [
            # Tie-Breaker Check
            (['b', 'a'],),
            (['x', 'z'],),
            (['red', 'green', 'blue'],),
            (['yes', 'no', 'yes', 'no'],),
        ]
    },    
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
    {
        "point": 0.25,
        "desc": "General Case (Numerical Attributes)",
        "tc": [
            # General Case
            ([[2.0], [4.0], [6.0], [8.0], [10.0]], ['yes', 'yes', 'no', 'no', 'no'], 2),
            ([[1], [0], [1], [0]], ['yes', 'no', 'yes', 'no'], 2),
            ([[8.0], [2.0], [10.0], [4.0], [6.0]], ['no', 'yes', 'no', 'yes', 'no'], 2),
        ]
    }, {
        "point": 0.25,
        "desc": "General Case (Categorical Attributes)",
        "tc": [
            # General Case
            ([['red'], ['blue'], ['red'], ['green'], ['blue']], ['yes', 'no', 'yes', 'no', 'no'], 2),
            ([['x'], ['y'], ['x'], ['z']], [2, 1, 2, 1], 2),
            ([['1'], ['2'], ['3'], ['1']], ['a', 'b', 'c', 'a'], 3),
        ]
    }, {
        "point": 0.25,
        "desc": "Edge Case: Mixed Attribute & Tie-Breaker Check",
        "tc": [
            # Tie-Breaker Check
            ([['a', 2.0], ['a', 4.0], ['b', 6.0], ['b', 8.0]], ['yes', 'yes', 'no', 'no'], 2),
            # Complex Tie-Breaker Check
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
    }, {
        "point": 0.25,
        "desc": "Edge Case: No Valid Split (Same X or Pure Node)",
        "tc": [
            # No Valid Split (Same X or Pure node)
            ([['same'], ['same'], ['same']], ['yes', 'yes', 'yes'], 1),
            ([[0], [0], [0]], ['yes', 'no', 'yes'], 2),
            ([[1], [2], [3], [4]], ['yes', 'yes', 'yes', 'yes'], 1),
            # Empty dataset
            # ([], [], 2),
        ]
    },    
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
    {
        "point": 0.25,
        "desc": "General Case (Numerical Attributes)",
        "tc": [
            # General Case (Numerical Attributes)
            ([[1], [2], [10], [12]], [0, 0, 1, 1], 0, 2, 2),
            ([[10], [2], [12], [3], [1]], [0, 1, 0, 1, 1], 0, 2, 2),
            ([[10], [2], [12], [3], [1]], [0, 1, 0, 1, 1], 0, 1, 2),
            ([[1], [2], [3], [4]], [0, 1, 2, 3], 0, 5, 4),   
        ]   
    }, {
        "point": 0.25,
        "desc": "General Case (Categorical Attributes)",
        "tc": [
            # General Case (Categorical Attributes)
            ([['Red'], ['Blue'], ['Red'], ['Green']], [0, 1, 0, 2], 0, 2, 3),
            ([['A'], ['B'], ['A'], ['C']], [0, 1, 0, 1], 0, 2, 2),  
        ]
    }, {
        "point": 0.25,
        "desc": "General Case (Mixed Attributes)",
        "tc": [
            # General Case (Mixed Attributes)
            ([[10, 'A'], [20, 'B'], [15, 'A'], [25, 'C']], [0, 1, 0, 1], 0, 3, 2),
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
    }, {
        "point": 0.25,
        "desc": "Edge Case: No Valid Split (Max Depth, Same X, Pure Node)",
        "tc": [
            # No Valid Split (Max Depth, Same X, Pure Node)
            ([[1], [2], [3]], [0, 1, 0], 2, 2, 2),
            ([[1], [2], [3]], [0, 1, 0], 0, 0, 2),
            ([[1], [2], [3]], [0, 0, 0], 0, 5, 2),
            ([[5], [5], [5]], [1, 1, 0], 0, 3, 2),
            ([[42]], [1], 0, 3, 1),
            # Empty X
            # ([], [0, 1, 0], 0, 5, 2),
            # ([], [], 0, 3, 0),
        ]
    }, 
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
    # ([X, y, depth, max_depth, n_unique_classes], x_instance)
    {
        "point": 0.5,
        "desc": "General Case",
        "tc": [
            # General Case
            ([[[3.0], [7.0], [2.0], [8.0]], [0, 1, 0, 1], 0, 1, 2], [2.5]),
            ([[[3.0], [7.0], [2.0], [8.0]], [0, 1, 0, 1], 0, 1, 2], [6.0]),
            ([[['apple'], ['banana'], ['apple'], ['orange']], [0, 1, 0, 1], 0, 1, 2], ['apple']),
            ([[['A', 10], ['B', 2], ['A', 12], ['C', 3]], [0, 1, 0, 1], 0, 1, 2], ['B', 5]),
            ([[[10, 'red'], [2, 'blue'], [12, 'green'], [3, 'red']], [0, 1, 0, 1], 0, 2, 2], [4, 'red']),
            ([[['red', 'circle'], ['blue', 'square'], ['red', 'triangle'], ['blue', 'circle']], [1, 0, 1, 0], 0, 2, 2], ['blue', 'triangle']),
            ([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], [0, 0, 1, 1], 0, 2, 2], [10.0, 12.0]),
            ([[[1, 'A'], [2, 'B'], [3, 'C'], [4, 'D']], [0, 1, 0, 1], 0, 2, 2], [2, 'B']),
        ]
    }, {
        "point": 0.5,
        "desc": "Edge Case: X_test = Continuous Split Value, Unseen Categorical Value (Use Default), or No Valid Split (Use Majority with Tie-Breaker)",
        "tc": [
            # X_test = Continuous Split Value
            ([[[2.0], [4.0], [6.0], [8.0]], [0, 0, 1, 1], 0, 1, 2], [5.0]),
            ([[[1.0], [1.2], [2.5]], [0, 0, 1], 0, 1, 2], [1.85]),
            # Unseen Categorical Value (Use Default)
            ([[['apple'], ['banana'], ['apple'], ['orange']], [0, 1, 0, 1], 0, 1, 2], ['grape']),
            ([[[10, 'A'], [2, 'B'], [12, 'A'], [3, 'B']], [0, 1, 0, 1], 0, 1, 2], [4, 'C']),
            ([[['cat'], ['dog'], ['cat'], ['mouse']], [0, 1, 0, 1], 0, 1, 2], ['elephant']),
            ([[[1, 'x', 10], [2, 'y', 20], [1, 'x', 30], [2, 'y', 40]], [0, 1, 0, 1], 0, 2, 2], [1, 'z', 25]),
            # No Valid Split (Use Majority with Tie-Breaker)
            ([[[5], [6], [7], [8]], [1, 1, 1, 1], 0, 1, 1], [9]),
            ([[[5], [5], [5], [5]], [2, 2, 1, 1], 0, 0, 2], [10]),
            ([[[1.0], [2.0], [3.0], [4.0]], [0, 1, 0, 1], 0, 1, 2], [2.5]),
        ]
    },    
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
    # (X, y, X_predict, max_depth)
    {
        "point": 0.5,
        "desc": "General Case",
        "tc": [
            # General Case
            ([[3.0], [7.0], [2.0], [8.0]], [0, 1, 0, 1], [[2.5], [6.0], [4.9], [5.1]], 1),
            ([['red'], ['blue'], ['red'], ['green']], [0, 1, 0, 1], [['red'], ['blue'], ['yellow']], 1),
            ([[1], [2], [6], [7]], [0, 0, 1, 1], [[1], [6], [4]], 3),
            (
                [[1, 'red'], [2, 'blue'], [3, 'red'], [4, 'green']], 
                [0, 0, 1, 1],
                [[1, 'red'], [2, 'blue'], [3, 'green'], [10, 'blue']], 
                1
            ),
        ]
    }, {
        "point": 0.5,
        "desc": "Edge Case: Unseen Categorical Value (Use Default), or No Valid Split (Use Majority with Tie-Breaker)",
        "tc": [
            # Unseen Categorical Value (Use Default)
            ([['cat'], ['dog'], ['dog'], ['cat']], [0, 1, 1, 0], [['cat'], ['dog'], ['rabbit'], ['horse'], ['snake']], 1),
            (
                [[10, 'apple'], [2, 'blue'], [12, 'banana'], [3, 'apple'], [4, 'orange']],
                [0, 1, 0, 1, 0],
                [[4, 'blue'], [6, 'banana'], [3, 'apple'], [1, 'orange'], [11, 'unknown']],
                2
            ),
            # No Valid Split (Use Majority with Tie-Breaker)
            ([[1.0], [2.0], [3.0]], [0, 0, 0], [[10.0], [20.0]], 5),
            ([[1.0], [1.0], [1.0], [1.0]], [0, 1, 0, 1], [[1.0], [2.0], [0.0]], 3),
            ([[1], [2], [3], [4]], [0, 1, 1, 1], [[10], [0], [2]], 0),
            # Empty List
            # ([], [], [[1.0], [2.0]], 2),
        ]
    },    
]

class DTClassifier_Solution:
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
# Additional Grader Functions
# -----------------------------
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

def grade_q26(student, solution, test_cases, check_fn, weight):
    total_point = 0
    feedbacks = []
    for tc_group in test_cases:
        max_point, desc, tc = tc_group["point"], tc_group["desc"], tc_group["tc"]
        max_point *= weight
        point = max_point
        fail = False
        try:
            for args, x_instance in tc:     
                tree = build_tree_recursive_solution(*args)
                output = solution(x_instance, tree)
                expected = student(x_instance, tree)

                if not check_fn(output, expected):
                    fail = True
                    break
            if fail:
                point = 0
            feedback = generate_feedback(point, max_point, desc)
        except Exception as e:
            point = 0
            feedback = generate_feedback(point, max_point, desc, error=str(e))

        total_point += point 
        feedbacks.append(feedback)
    return total_point, feedbacks
    
# -----------------------------
# Autograde Workflow
# -----------------------------
TASKS = {
    "1.1": ("function", "calculate_regionrss", calculate_regionrss_solution, float_assert, TC_1_1),
    "1.2": ("class", "DTRegressor", DTRegressor_solution, list_float_assert, TC_1_2),
    "2.1": ("function", "compute_entropy", compute_entropy_solution, float_assert, TC_2_1), 
    "2.2": ("function", "information_gain", information_gain_solution, float_assert, TC_2_2), 
    "2.3": ("function", "majority_class", majority_class_solution, default_assert, TC_2_3), 
    "2.4": ("function", "find_best_split", find_best_split_solution, q24_assert, TC_2_4),
    "2.5": ("function", "build_tree_recursive", build_tree_recursive_solution, default_assert, TC_2_5),
    "2.6": ("2_6", "predict_one_instance", predict_one_instance_solution, float_assert, TC_2_6),
    "2.7": ("class", "DTClassifier", DTClassifier_Solution, list_float_assert, TC_2_7),
    "3": ("model", "train_model", None, ACCURACY_THRESHOLD, load_iris())
}

def autograde_folder(folder):
    all_scores = []
    fails = []

    for filename in tqdm(os.listdir(folder)):
        if not filename.endswith(".ipynb"):
            continue

        nb_path = os.path.join(folder, filename)
        module_path = nb_path.replace(".ipynb", ".py")
        file = filename[:-6]

        if file in {"sohkaile_132704_7279504_Soh_Kai_Le_A0273076B_assignment2-1",
                    "muhammadzafranshahbmahadhir_LATE_28061_7332529_Muhammad Zafranshah Bin Mahadhir_A0230456L_assignment2"}:
            continue
        try:
            # Convert notebook -> module
            notebook_to_module(nb_path, module_path)
            module_name = file.replace(".ipynb", "")
            module = import_module_safe(module_name, module_path)
        except Exception as e:
            # Catch notebooks that fail to run
            fails.append(f"{file}: {str(e)}")
            scores = [file] + ["X"] * (len(TASKS.keys()) + 1)
            all_scores.append(scores)
            if os.path.exists(module_path):
                os.remove(module_path)
            continue

        # Grade function
        total_score = 0
        feedbacks = ""
        name = file.replace("-", "_").split("_")[0]
        student_number_search = re.search(r"(A\d{7}[A-Z])", file)
        student_number = student_number_search.group(1) if student_number_search else ""
        scores = [file, student_number, name]
        for task_num, (type, task, solution_fn, check_fn, test_cases) in TASKS.items():
            weight = GRADE_DISTRIBUTION[task_num]
            student_fn = getattr(module, task)
            if type in "function":
                score, feedback_list = grade_function(student_fn, solution_fn, test_cases, check_fn, weight)
            elif type in "class":
                score, feedback_list = grade_class(student_fn, solution_fn, test_cases, check_fn, weight)
            elif type == "model":
                score, feedback_list = grade_model(student_fn, test_cases, check_fn, weight) 
            elif type == "2_6":
                score, feedback_list = grade_q26(student_fn, solution_fn, test_cases, check_fn, weight)
            feedback = "; ".join(feedback_list)
            scores.extend([score, feedback])
            total_score += score
            feedbacks += feedback + "\n"
            # Inject the correct definition for subsequent questions
            module.__dict__[task] = solution_fn

        scores.extend([total_score, feedbacks])
        all_scores.append(scores)

        os.remove(module_path)

    print(f"Failed to compile: {fails}")
    col_names = sorted(list(TASKS.keys()) + list(map(lambda x: f"{x} Feedback", TASKS.keys())))
    score_columns = ["filename", "student_number", "name"] + col_names + ["total", "all_feedbacks"]
    return pd.DataFrame(all_scores, columns=score_columns), fails

# -----------------------------
# Run autograder
# -----------------------------
if __name__ == "__main__":
    score_report, fails = autograde_folder(NOTEBOOK_FOLDER)
    save_to_csv(score_report, SCORE_CSV)
    save_fails(fails, FAILS_TXT)
    print(f"Grading completed! Results saved to {SCORE_CSV}")
