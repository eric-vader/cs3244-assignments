import math
import os
import re
import sys
from collections import Counter

import pandas as pd
from sklearn.datasets import fetch_lfw_people
from tqdm import tqdm

from utils.grading_util import *
from utils.import_export_util import *

# -----------------------------
# Configuration
# -----------------------------
sys.dont_write_bytecode = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER = "submissions"
NOTEBOOK_FOLDER = os.path.join(BASE_DIR, FOLDER)
SCORE_CSV = os.path.join(BASE_DIR, f"a1_grades_{FOLDER}.csv")
FAILS_TXT = os.path.join(BASE_DIR, f"a1_fails.txt")
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
            (['blue', 'green', 'green', 'blue'], ),
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
DATASET = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
ACCURACY_THRESHOLD = 0.5
    
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
    "5": ("model", "train_model", None, ACCURACY_THRESHOLD, DATASET)
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
            feedback = "; ".join(feedback_list)
            scores.extend([score, feedback])
            total_score += score
            feedbacks += feedback + "\n"
            # Inject the correct definition for subsequent questions
            module.__dict__[task] = solution_fn

        scores.append(total_score)
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
