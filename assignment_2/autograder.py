import os
import nbformat
from nbconvert import PythonExporter
import importlib.util
import sys
import math
from collections import Counter
import pandas as pd
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
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
GRADE_DISTRIBUTION = [1, 1, 1, 1, 2, 2, 1]

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
"""
# -----------------------------
# Task 4.2
# -----------------------------
TC_4_2 = [
    # (X_train, y_train, X_test, k)
    # General test case
    ([[1], [2], [3], [4], [5]], [10.0, 20.0, 30.0, 40.0, 50.0], [[2.5], [4.5]], 3),
    ([[1, 1], [2, 2], [3, 3]], [10.0, 20.0, 30.0], [[1.2, 1.2], [2.8, 2.8], [0.5, 0.5]], 1),
    ([[1, 5], [2, 1], [3, 6], [4, 2], [5, 7]], [50.0, 10.0, 60.0, 20.0, 70.0], [[2.5, 3.0]], 2),
    ([[10], [20], [30]], [100.0, 200.0, 300.0], [[15]], 3)
]

class KNNRegressor_solution:
    def __init__(self, k=3):
        self.k = k
        self.training_data = [] 

    def fit(self, X, y):
        self.training_data = list(zip(X, y))

    def predict(self, X_test):
        predictions = []
        # Separate training features and target values from the stored training_data
        # self.training_data is a list of (feature_vector, target_value) tuples
        X_train_fit = [item[0] for item in self.training_data]
        y_train_fit = [item[1] for item in self.training_data]

        for test_point in X_test:
            # Call the knn_regression function for each test point
            predicted_value = knn_regression_solution(X_train_fit, y_train_fit, test_point, self.k)
            predictions.append(predicted_value)
        return predictions
    
# -----------------------------
# Task 3
# -----------------------------
ACCURACY_THRESHOLD = 0.5

def grade_q5(student_fn):
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    X = lfw_people.data  # Flattened images
    y = lfw_people.target

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    try:
        model = student_fn(X_train, y_train)
        predictions = model.predict(X_test)
        if len(predictions) != len(X_test):
            return 0
        accuracy_score = model.score(X_test, y_test)
        return 1 if accuracy_score > ACCURACY_THRESHOLD else 0
    except Exception:
        return 0
"""
# -----------------------------
# Generic Grader Functions
# -----------------------------
def default_assert(output, expected):
    return output == expected

def float_assert(output, expected, tolerance=1e-5):
    return abs(output - expected) < tolerance

def list_float_assert(output, expected, tolerance=1e-5):
    return all(abs(o - e) < tolerance for o, e in zip(output, expected))

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
    
# -----------------------------
# Autograde Workflow
# -----------------------------
TASKS = [
    # (type, name, solution, check_type, test_cases, grade_weight)
    ("function", "calculate_regionrss", calculate_regionrss_solution, float_assert, TC_1_1, GRADE_DISTRIBUTION[0]),
    ("class", "DecisionTreeRegressor", DecisionTreeRegressor_solution, list_float_assert, TC_1_2, GRADE_DISTRIBUTION[1]),
    ("function", "compute_entropy", compute_entropy_solution, float_assert, TC_2_1, GRADE_DISTRIBUTION[2]), 
    ("function", "information_gain", information_gain_solution, float_assert, TC_2_2, GRADE_DISTRIBUTION[3]), 
    ("function", "majority_class", majority_class_solution, default_assert, TC_2_3, GRADE_DISTRIBUTION[4]), 
    # ("class", "KNNRegressor", KNNRegressor_solution, list_float_assert, TC_4_2, GRADE_DISTRIBUTION[5]),
    # ("q5", "train_model", None, None, None, GRADE_DISTRIBUTION[6])
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
    return pd.DataFrame(rows, columns=["student_number", "1.1", "1.2", "2.1", "2.2", "2.3", "total"])

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
