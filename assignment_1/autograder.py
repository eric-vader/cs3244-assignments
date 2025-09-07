import ast
import importlib.util
import io
import math
import os
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
# Configuration
# -----------------------------
sys.dont_write_bytecode = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_FOLDER = os.path.join(BASE_DIR, "submissions")
SCORE_CSV = os.path.join(BASE_DIR, "a1_grades.csv")
FEEDBACK_CSV = os.path.join(BASE_DIR, "a1_feedbacks.csv")
FAILS_TXT = os.path.join(BASE_DIR, "a1_fails.txt")
GRADE_DISTRIBUTION = {
    "1": 1.0,
    "2": 1.0,
    "3.1": 1.0,
    "3.2": 1.0,
    "4.1": 2.0,
    "4.2": 2.0,
    "5": 2.0,
}

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
# Task 1
# -----------------------------
TC_1 = [
    # (vec_p, vec_q)
    {
        "point": 0.5,
        "desc": "General Case",
        "tc": [
            # General Case
            ([1, 2, 3], [4, 5, 6]),
            ([0, 0], [0, 0]),
            ([1.5, 2.5], [3.0, 4.0]),
            ([1, 2, 3], [1, 2, 3]),
            ([5], [2]),
            # Large Dimension
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]),
            (list(range(100)), list(range(100, 200))),
        ]
    }, {
        "point": 0.5,
        "desc": "Edge Case: Negative Input, Tuple Input or Empty List",
        "tc": [
            # Negative Input
            ([-1, -2], [-3, -4]),
            ([-1, 2], [3, -5]),
            # Tuple Input
            ((1, 2), (1, 2)),
            # Empty List
            ([], []),
        ]
    },    
]

def euclidean_distance_solution(vec_p, vec_q):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_p, vec_q)))

# -----------------------------
# Task 2
# -----------------------------
TC_2 = [
    # (training_data, test_point, k)
    {
        "point": 0.5,
        "desc": "General Case",
        "tc": [
            # General Case
            ([([1, 2], 'A'),([2, 3], 'B'),([3, 4], 'A'),([5, 5], 'B')], [1.5, 2.5], 2),
            ([([1, 2], 'A'),([2, 3], 'B'),([3, 4], 'A'),([5, 5], 'B')], [4, 4], 1),
            ([([1, 2], 'A'),([2, 3], 'B'),([3, 4], 'A'),([5, 5], 'B')], [0, 0], 3),
            ([([0, 0], 'X'),([1, 1], 'Y'),([2, 2], 'Y'),([3, 3], 'X')], [1.5, 1.5], 2),
            ([([5, 5], 'Cat'),([1, 2], 'Dog'),([6, 6], 'Cat'),([2, 3], 'Dog')], [5.5, 5.5], 2),
            ([([10, 10], 'Apple'),([1, 1], 'Banana'),([2, 2], 'Banana'),([8, 8], 'Apple')], [2.5, 2.5], 2),
        ]
    }, {
        "point": 0.5,
        "desc": "Edge Case: Tie-Breaker Check or Train Point = Test Point",
        "tc": [
            # Tie-Breaker Check
            ([([1, 2], True), ([3, 4], False)], [2.0, 3.0], 2),
            ([([1, 2], 'A'),([1, 2], 'B')], [1, 2], 2),
            # Train point same as test point
            ([([0, 0], 'X'),([3, 0], 'Y')], [0, 0], 1),
            # Different Type Labels (ommited)
            # ([([1, 1], 'A')], [1, 1], 1),
            # ([([1, 2], 1),([3, 4], 'B')], [2.5, 2.5], 1),
        ]
    },    
]

def get_k_nearest_neighbors_solution(training_data, test_point, k):
        distances = []
        for feature_vector, label in training_data:
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(feature_vector, test_point)))
            distances.append((dist, label))
        distances.sort(key=lambda x: x[0])
        k_neighbors = [label for _, label in distances[:k]]
        return k_neighbors

# -----------------------------
# Task 3.1
# -----------------------------
TC_3_1 = [
    # (neighbours,)
    {
        "point": 0.5,
        "desc": "General Case",
        "tc": [
            # General Case
            (['A', 'B', 'A'], ),
            (['B', 'B', 'A'], ),
            (['A', 'A', 'A'], ),
            (['dog', 'cat', 'dog', 'bird', 'dog'], ),
            (['blue', 'green', 'green', 'blue'], ),
            (['z'], ),
        ]
    }, {
        "point": 0.5,
        "desc": "Edge Case: Non-String Labels or Tie-Breaker Check",
        "tc": [
            # Non-String Labels
            ([1.2, 41.5, 6.24, 21.9, 1.2, 0.0, 2.0, 6.24], ),
            ([False, True, False, False, False, True], ),
            # Tie-Breaker Check
            ([3, 2, 2, 3], ),
            (['B', 'B', 'A', 'A'], ),
        ]
    },    
]

def knn_majority_vote_solution(neighbors):
    count = Counter(neighbors)
    most_common_label = count.most_common(1)[0][0]
    return most_common_label
    
# -----------------------------
# Task 3.2
# -----------------------------
TC_3_2 = [
    # (X_train, y_train, x_query, k)
    {
        "point": 0.5,
        "desc": "General Case",
        "tc": [
            # General Case
            ([[1], [2], [3], [4], [5]], [1.1, 1.9, 3.0, 3.9, 5.1], [2.5], 2),
            ([[1], [2], [3]], [1, 2, 3], [2.1], 1),
            ([[1], [2], [3]], [1, 2, 3], [2], 3),
            ([[1, 2], [2, 3], [3, 4]], [10, 20, 30], [2, 2.5], 2),
            ([[0, 0, 0], [1, 1, 1], [2, 2, 2]], [0, 3, 6], [1.5, 1.5, 1.5], 2),
            ([[0], [2]], [1, 3], [1], 2),
            ([[-5.2, -12.5], [-1.1, -4.2], [-2.8, -1.9]], [-3.1, -6.2, -9.5], [-1.2, -2.1], 1),
            ([[1], [2], [3]], [1.0, 2.0, 3.0], [2], 2),

        ]
    }, {
        "point": 0.5,
        "desc": "Edge Case: Boolean Labels, Tie-Breaker Check, Train Point = Test Point, or Duplicate Train Data",
        "tc": [
            # Tie-Breaker Check
            ([[1], [3]], [2, 4], [2], 2),
            # Train Point = Test Point
            ([[1], [2], [3]], [1, 2, 3], [2], 1),
            # Boolean labels
            ([[1], [2], [3]], [True, False, True], [2], 2),
            # Duplicate training data
            ([[1], [2], [2], [3]], [1, 2, 2, 3], [2], 3),
        ]
    },    
]

def knn_regression_solution(X_train, y_train, x_query, k):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance_solution(X_train[i], x_query)
        distances.append((dist, y_train[i]))

    distances.sort()
    k_nearest = distances[:k]
    if not k_nearest: # Handle case where no neighbors are found (e.g., empty X_train)
        return 0.0 
    prediction = sum(val for _, val in k_nearest) / len(k_nearest) # fixed from k to len(k_nearest)
    return prediction
    
# -----------------------------
# Task 4.1
# -----------------------------
TC_4_1 = [
    # (X_train, y_train, X_test, k)
    {
        "point": 0.5,
        "desc": "General Case",
        "tc": [
            # General Case
            ([[1, 2], [2, 3], [3, 4], [5, 5]], ['A', 'B', 'A', 'B'], [[1.5, 2.5], [4, 4]], 3),
            ([[1, 1], [2, 2], [3, 3], [4, 4]], ['A', 'A', 'B', 'B'], [[1.5, 1.5], [3.5, 3.5]], 1),
            ([[i, i*2] for i in range(10)], ['A' if i % 2 == 0 else 'B' for i in range(10)], [[0.5, 1.0], [4.2, 8.5], [7.8, 15.1]], 5),
            ([[0, 0], [1, 1], [0, 1], [1, 0]], ['A', 'A', 'B', 'B'], [[0.5, 0.5], [0.1, 0.9], [0.9, 0.1]], 2),
            ([[1.2, 2.1, 3.6], [4.1, 5.9, 6.2], [2.5, 3.1, 4.9], [-5, 6, -7], [1.3, 7.4, 0.5]], ['A', 'B', 'A', 'B', 'A'], [[2, 3, 3.5], [4.5, 5.5, 6.5]], 3),
        ]
    }, {
        "point": 0.5,
        "desc": "Edge Case: Tie-Breaker Check or Train Point = Test Point",
        "tc": [
            # Train Point = Test Point
            ([[1, 1], [1, 1], [1, 1]], ['A', 'A', 'A'], [[1, 1]], 3),
            # Tie-Breaker Check
            ([[0, 0], [1, 1], [2, 2], [3, 3]], ['A', 'B', 'B', 'A'], [[1.5, 1.5]], 4),
            # Non-string labels (omitted)
            # ([[1, 2], [3, 4]], ['A', 5], [[2, 3]], 1),
        ]
    },    
]

class KNNClassifier_solution:
    def __init__(self, k=3):
        self.k = k
        self.training_data = [] 

    def fit(self, X, y):
        self.training_data = list(zip(X, y))

    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            neighbors = get_k_nearest_neighbors_solution(self.training_data, test_point, self.k)
            predicted_label = knn_majority_vote_solution(neighbors)
            predictions.append(predicted_label) 
        return predictions
    
# -----------------------------
# Task 4.2
# -----------------------------
TC_4_2 = [
    # (X_train, y_train, X_test, k)
    {
        "point": 1,
        "desc": "General Case",
        "tc": [
            # General Case
            ([[1], [2], [3], [4], [5]], [10.0, 20.0, 30.0, 40.0, 50.0], [[2.5], [4.5]], 3),
            ([[1, 1], [2, 2], [3, 3]], [10.0, 20.0, 30.0], [[1.2, 1.2], [2.8, 2.8], [0.5, 0.5]], 1),
            ([[1, 5], [2, 1], [3, 6], [4, 2], [5, 7]], [50.0, 10.0, 60.0, 20.0, 70.0], [[2.5, 3.0]], 2),
            ([[10], [20], [30]], [100.0, 200.0, 300.0], [[15]], 3)
        ]
    }
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
# Task 5
# -----------------------------
ACCURACY_THRESHOLD_HIGH = 0.5

# -----------------------------
# Generic Grader Functions
# -----------------------------
def default_assert(output, expected):
    return output == expected

def float_assert(output, expected, tolerance=1e-5):
    return abs(output - expected) < tolerance

def list_float_assert(output, expected, tolerance=1e-5):
    return len(output) == len(expected) and all(abs(o - e) < tolerance for o, e in zip(output, expected))

def generate_feedback(point, max_point, desc, error=None):
    if not error:
        return f"[{point}/{max_point}] {desc}"
    else:
        return f"[{point}/{max_point}] Error ({error}) in {desc}"

def grade(type, student, solution, test_cases, check_fn, weight, module=None):
    total_point = 0
    feedbacks = []
    for tc_group in test_cases:
        max_point, desc, tc = tc_group["point"], tc_group["desc"], tc_group["tc"]
        max_point *= weight
        point = max_point
        fail = False
        try:
            if type == "function":
                for args in tc:
                    output = student(*args)
                    expected = solution(*args)
                    if not check_fn(output, expected):
                        fail = True
                        break
            elif type == "class":
                for X_train, y_train, X_test, class_arg in tc:     
                    module.__dict__["X_train"] = X_train
                    module.__dict__["y_train"] = y_train

                    output_class = student(class_arg)
                    output_class.fit(X_train, y_train)
                    output = output_class.predict(X_test)
                    expected_class = solution(class_arg)
                    expected_class.fit(X_train, y_train)
                    expected = expected_class.predict(X_test)
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

def grade_model(student_fn, weight):
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    X = lfw_people.data  # Flattened images
    y = lfw_people.target
    max_point = weight
    # Split into training and test sets
        
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        f = io.StringIO()
        with redirect_stdout(f):
            model = student_fn(X_train, y_train)
            accuracy_score = model.score(X_test, y_test)

        if accuracy_score > ACCURACY_THRESHOLD_HIGH:
            point = max_point
            feedback = generate_feedback(point, max_point, "Performance is above threshold")
        else:
            point = 0
            feedback = generate_feedback(point, max_point, "Performance is below threshold")

    except Exception as e:
        point = 0
        if str(e) == "'NoneType' object has no attribute 'score'":
            feedback = generate_feedback(point, max_point, "No model trained")
        elif "not defined" in str(e) and ("X_test" in str(e) or "X" in str(e)):
            feedback = generate_feedback(point, max_point, "Accessed test case")
        else:
            feedback = generate_feedback(point, max_point, "Evaluation", error=str(e))
    return point, [feedback]
    
# -----------------------------
# Autograde Workflow
# -----------------------------
TASKS = {
    "1": ("function", "euclidean_distance", euclidean_distance_solution, float_assert, TC_1),
    "2": ("function", "get_k_nearest_neighbors", get_k_nearest_neighbors_solution, default_assert, TC_2),
    "3.1": ("function", "knn_majority_vote", knn_majority_vote_solution, default_assert, TC_3_1), 
    "3.2": ("function", "knn_regression", knn_regression_solution, float_assert, TC_3_2), 
    "4.1": ("class", "KNNClassifier", KNNClassifier_solution, default_assert, TC_4_1), 
    "4.2": ("class", "KNNRegressor", KNNRegressor_solution, list_float_assert, TC_4_2),
    "5": ("model", "train_model", None, None, None)
}

def autograde_folder(folder):
    all_scores = []
    fails = []
    for filename in tqdm(os.listdir(folder)):
        if not filename.endswith(".ipynb"):
            continue

        nb_path = os.path.join(folder, filename)
        module_path = nb_path.replace(".ipynb", ".py")
        student_number = filename[:-6]
        
        try:
            # Convert notebook -> module
            notebook_to_module(nb_path, module_path)
            module_name = filename.replace(".ipynb", "")
            module = import_module_safe(module_name, module_path)
        except Exception as e:
            # Catch notebooks that fail to run
            fails.append(f"{student_number}: {str(e)}")
            scores = [student_number] + ["X"] * (len(TASKS.keys()) + 1)
            all_scores.append(scores)
            if os.path.exists(module_path):
                os.remove(module_path)
            continue

        # Grade function
        total_score = 0
        scores = [student_number]
        for task_num, (type, task, solution_fn, check_fn, test_cases) in TASKS.items():
            weight = GRADE_DISTRIBUTION[task_num]
            student_fn = getattr(module, task)
            if type == "function":
                score, feedback = grade(type, student_fn, solution_fn, test_cases, check_fn, weight)
            elif type == "class":
                score, feedback = grade(type, student_fn, solution_fn, test_cases, check_fn, weight, module)
            elif type == "model":
                score, feedback = grade_model(student_fn, weight) 
            scores.append(score)
            scores.append("; ".join(feedback))
            total_score += score
        scores.append(total_score)
        all_scores.append(scores)

        os.remove(module_path)

    print(f"Failed to compile: {fails}")
    col_names = sorted(list(TASKS.keys()) + list(map(lambda x: f"{x} Feedback", TASKS.keys())))
    score_columns = ["student_number"] + col_names + ["total"]
    return pd.DataFrame(all_scores, columns=score_columns), fails

# -----------------------------
# Save report to CSV
# -----------------------------
def save_to_csv(report, csv_file):
    report.to_csv(csv_file, index = False)

def save_fails(fails, txt_file):
    with open(txt_file, "w") as f:
        for item in fails:
            f.write(f"{item}\n")


# -----------------------------
# Run autograder
# -----------------------------
if __name__ == "__main__":
    score_report, fails = autograde_folder(NOTEBOOK_FOLDER)
    save_to_csv(score_report, SCORE_CSV)
    save_fails(fails, FAILS_TXT)
    print(f"Grading completed! Results saved to {SCORE_CSV}")
