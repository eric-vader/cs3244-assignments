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
OUTPUT_CSV = os.path.join(BASE_DIR, "a1_grades.csv")
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
# Task 1
# -----------------------------
TC_1 = [
    # (vec_p, vec_q)
    # General test case
    ([1, 2, 3], [4, 5, 6]),
    ([0, 0], [0, 0]),
    ([1.5, 2.5], [3.0, 4.0]),
    ([1, 2, 3], [1, 2, 3]),
    ([5], [2]),

    # Large Dimensions
    ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]),
    (list(range(100)), list(range(100, 200))),

    # Negative value
    ([-1, -2], [-3, -4]),
    ([-1, 2], [3, -5]),

    # Empty List
    ([], []),
    
    # List and tuple
    ([1, 2], (1, 2))
]

def euclidean_distance_solution(vec_p, vec_q):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_p, vec_q)))

# -----------------------------
# Task 2
# -----------------------------
TC_2 = [
    # (training_data, test_point, k)
    # General test case
    ([([1, 2], 'A'),([2, 3], 'B'),([3, 4], 'A'),([5, 5], 'B')], [1.5, 2.5], 2),
    ([([1, 2], 'A'),([2, 3], 'B'),([3, 4], 'A'),([5, 5], 'B')], [4, 4], 1),
    ([([1, 2], 'A'),([2, 3], 'B'),([3, 4], 'A'),([5, 5], 'B')], [0, 0], 3),
    ([([0, 0], 'X'),([1, 1], 'Y'),([2, 2], 'Y'),([3, 3], 'X')], [1.5, 1.5], 2),
    ([([5, 5], 'Cat'),([1, 2], 'Dog'),([6, 6], 'Cat'),([2, 3], 'Dog')], [5.5, 5.5], 2),
    ([([10, 10], 'Apple'),([1, 1], 'Banana'),([2, 2], 'Banana'),([8, 8], 'Apple')], [2.5, 2.5], 2),

    # Non-string labels
    ([([1, 2], 1),([3, 4], 'B')], [2.5, 2.5], 1),
    
    # Order-preservation check
    ([([1, 2], True), ([3, 4], False)], [2.0, 3.0], 2),
    ([([1, 2], 'A'),([1, 2], 'B')], [1, 2], 2),

    # Training point same as test point
    ([([0, 0], 'X'),([3, 0], 'Y')], [0, 0], 1),
    ([([1, 1], 'A')], [1, 1], 1),
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
    # neighbours
    # General test case
    (['A', 'B', 'A'], ),
    (['B', 'B', 'A'], ),
    (['A', 'A', 'A'], ),
    (['dog', 'cat', 'dog', 'bird', 'dog'], ),
    (['blue', 'green', 'green', 'blue'], ),
    (['z'], ),

    # Non-string labels
    ([1.2, 41.5, 6.24, 21.9, 1.2, 0.0, 2.0, 6.24], ),
    ([False, True, False, False, False, True], ),

    # Order-preservation check
    ([3, 2, 2, 3], ),
    (['B', 'B', 'A', 'A'], ),
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
    # General test case
    ([[1], [2], [3], [4], [5]], [1.1, 1.9, 3.0, 3.9, 5.1], [2.5], 2),
    ([[1], [2], [3]], [1, 2, 3], [2.1], 1),
    ([[1], [2], [3]], [1, 2, 3], [2], 3),
    ([[1, 2], [2, 3], [3, 4]], [10, 20, 30], [2, 2.5], 2),
    ([[0, 0, 0], [1, 1, 1], [2, 2, 2]], [0, 3, 6], [1.5, 1.5, 1.5], 2),
    ([[0], [2]], [1, 3], [1], 2),
    ([[-5.2, -12.5], [-1.1, -4.2], [-2.8, -1.9]], [-3.1, -6.2, -9.5], [-1.2, -2.1], 1),
    ([[1], [2], [3]], [1.0, 2.0, 3.0], [2], 2),

    # Training point same as test point
    ([[1], [2], [3]], [1, 2, 3], [2], 1),

    # Non-number labels
    ([[1], [2], [3]], [True, False, True], [2], 2),

    # Order-preservation check
    ([[1], [3]], [2, 4], [2], 2),

    # Duplicate training data
    ([[1], [2], [2], [3]], [1, 2, 2, 3], [2], 3),
    
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
    # General test case
    ([[1, 2], [2, 3], [3, 4], [5, 5]], ['A', 'B', 'A', 'B'], [[1.5, 2.5], [4, 4]], 3),
    ([[1, 1], [2, 2], [3, 3], [4, 4]], ['A', 'A', 'B', 'B'], [[1.5, 1.5], [3.5, 3.5]], 1),
    ([[i, i*2] for i in range(10)], ['A' if i % 2 == 0 else 'B' for i in range(10)], [[0.5, 1.0], [4.2, 8.5], [7.8, 15.1]], 5),
    ([[0, 0], [1, 1], [0, 1], [1, 0]], ['A', 'A', 'B', 'B'], [[0.5, 0.5], [0.1, 0.9], [0.9, 0.1]], 2),
    ([[1.2, 2.1, 3.6], [4.1, 5.9, 6.2], [2.5, 3.1, 4.9], [-5, 6, -7], [1.3, 7.4, 0.5]], ['A', 'B', 'A', 'B', 'A'], [[2, 3, 3.5], [4.5, 5.5, 6.5]], 3),

    # Non-string labels
    ([[1, 2], [3, 4]], ['A', 5], [[2, 3]], 1),

    # Training point same as test point
    ([[1, 1], [1, 1], [1, 1]], ['A', 'A', 'A'], [[1, 1]], 3),

    # Order-preservation check
    ([[0, 0], [1, 1], [2, 2], [3, 3]], ['A', 'B', 'B', 'A'], [[1.5, 1.5]], 4),
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
# Task 5
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

# -----------------------------
# Autograde Workflow
# -----------------------------
TASKS = [
    ("function", "euclidean_distance", euclidean_distance_solution, float_assert, TC_1, GRADE_DISTRIBUTION[0]),
    ("function", "get_k_nearest_neighbors", get_k_nearest_neighbors_solution, default_assert, TC_2, GRADE_DISTRIBUTION[1]),
    ("function", "knn_majority_vote", knn_majority_vote_solution, default_assert, TC_3_1, GRADE_DISTRIBUTION[2]), 
    ("function", "knn_regression", knn_regression_solution, float_assert, TC_3_2, GRADE_DISTRIBUTION[3]), 
    ("class", "KNNClassifier", KNNClassifier_solution, default_assert, TC_4_1, GRADE_DISTRIBUTION[4]), 
    ("class", "KNNRegressor", KNNRegressor_solution, list_float_assert, TC_4_2, GRADE_DISTRIBUTION[5]),
    ("q5", "train_model", None, None, None, GRADE_DISTRIBUTION[6])
]

def autograde_folder(folder):
    rows = []
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
        module = import_module_from_path(module_name, module_path)
        
        # Grade function
        total_score = 0

        row = [student_number]
        for type, task, solution_fn, check_fn, test_cases, weight in TASKS:
            student_fn = getattr(module, task)
            if type == "function":
                score = grade_function(student_fn, solution_fn, test_cases, check_fn) * weight
            elif type == "class":
                score = grade_class(student_fn, solution_fn, test_cases, check_fn) * weight
            elif type == "q5":
                score = grade_q5(student_fn) * weight

            row.append(score)
            total_score += score
        row.append(total_score)
        rows.append(row)

        try:
            os.remove(module_path)
        except OSError as e:
            print(f"Error deleting {module_path}: {e}")
    return pd.DataFrame(rows, columns=["student_number", "1", "2", "3.1", "3.2", "4.1", "4.2", "5", "total"])

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
