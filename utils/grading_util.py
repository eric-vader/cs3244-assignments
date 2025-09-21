import io
from contextlib import redirect_stdout

from sklearn.model_selection import train_test_split

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

def grade_function(student_fn, solution_fn, test_cases, check_fn, weight):
    total_point = 0
    feedbacks = []
    for tc_group in test_cases:
        max_point, desc, tc = tc_group["point"], tc_group["desc"], tc_group["tc"]
        max_point *= weight
        point = max_point
        fail = False
        try:
            for args in tc:
                output = student_fn(*args)
                expected = solution_fn(*args)
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

def grade_class(student_class, solution_class, test_cases, check_fn, weight):
    total_point = 0
    feedbacks = []
    for tc_group in test_cases:
        max_point, desc, tc = tc_group["point"], tc_group["desc"], tc_group["tc"]
        max_point *= weight
        point = max_point
        fail = False
        try:
            for X_train, y_train, X_test, class_arg in tc:   
                output_class = student_class(class_arg)
                output_class.fit(X_train, y_train)
                output = output_class.predict(X_test)
                expected_class = solution_class(class_arg)
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

def grade_model(student_fn, dataset, accuracy_threshold, weight):
    X, y = dataset.data, dataset.target
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

        if accuracy_score > accuracy_threshold:
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