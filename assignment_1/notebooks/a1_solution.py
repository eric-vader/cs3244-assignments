#!/usr/bin/env python
# coding: utf-8

# # Assignment 1: k-Nearest Neighbors

# ## Overview
# 
# In this lab, we'll implement KNN step-by-step:
# 
# 1. **Euclidean Distance:** Measure similarity between data points.
# 2. **Find Neighbors:** Select the 'K' closest points to a query point.
# 3. **Predict (Classification):** Use majority vote from neighbors.
# 4. **Predict (Regression):** Use the average of neighbors' values.
# 5. **Build KNN Class:** Combine everything into a reusable class.
# 6. **Practical**: Train a KNN classifier on the training dataset using scikit-learn.
# 
# By the end, you'll understand how KNN works and how it makes predictions based on distance. Let’s dive in!

# Please note the following:
# 1. Fill in your name and student number at the top of the ipynb file.
# 2. The parts you need to implement are clearly marked with the following:
# 
#     ```
#     """ YOUR CODE STARTS HERE """
# 
#     """ YOUR CODE ENDS HERE """
#     ```
#     
#     , and you should write your code in between the above two lines.
# 3. Note that for each part, you are not allowed to use numpy. If you are unsure whether a particular function is allowed, feel free to ask any of the TAs.
# 
# 
# ### Submission Instructions
# Items to be submitted:
# * **Source code (lab4.py).** This is where you fill in all your code.
# * **Report (report.pdf).** This should describe your implementation and be no more than one page.
# Please clearly indicate your name and student number (the one that looks like A1234567X) in the report as well as the top of your source code. Zip the two files together and name it in the following format: A1234567X.zip (replace with your student number). Opening the zip file should show:
# 
# Submit your assignment by **15 April 2025, 2359HRS** to Canvas. 25% of the total score will be deducted for each day of late submission.
# 

# In[1]:


import math 


# ## Task 1 - Compute Euclidean Distance

# To find the nearest neighbors, we first need a distance measure to determine how **close** two data points are.
# 
# A possible distance measure is **Euclidean distance**—the straight-line distance between points $\mathbf{p} = [p_1, p_2, ..., p_m]$ and $\mathbf{q} = [q_1, q_2, ..., q_m]$ in $m$ dimensions:
# 
# $$
# d_\text{euclidean}(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{m} (p_i - q_i)^2}
# $$
# 
# Implement the function calculating euclidean distance using `math.sqrt()`, without the use of `numpy`.
# 
# Need to work for any dimension sizes 

# In[2]:


#[TASK 1]
def euclidean_distance(vec_p, vec_q):
    """
    Args:
        vec_p: Vector p, with a list of m elements.
        vec_q: Vector q, with a list of m elements.
    Returns:
        A float representing the Euclidean distance between the two vectors
    """

    # TODO: Write a function to compute distance between two vectors
    # Call the math.sqrt() function to compute the square root of the sum of squared differences  
    # If you see this error - NameError: name 'math' is not defined, make sure you've run the previous cell where we imported the math module

    """ YOUR CODE STARTS HERE """    
    squared_differences = []
    for i in range(len(vec_p)):
        difference = vec_p[i] - vec_q[i]
        squared_difference = difference ** 2
        squared_differences.append(squared_difference)

    sum_of_squares = sum(squared_differences)
    distance = math.sqrt(sum_of_squares)
    return distance
    """ YOUR CODE ENDS HERE """


#[TESTCASES 1.1]
# If you pass the following test case, there will be no output!
assert math.isclose(euclidean_distance([1, 2, 3], [4, 5, 6]), 5.196152422706632, rel_tol=1e-5)
assert math.isclose(euclidean_distance([5.5, 5.5],[5.5, 5.5]), 0.0, rel_tol=1e-5)
assert math.isclose(euclidean_distance([],[]), 0.0, rel_tol=1e-5)
print('All test cases passed!') 
#[/TESTCASES]


# ## Task 2 - Get the K Nearest Neighbors

# Now that we can measure distance, let’s use that to find the ```k``` closest training points to a given test point. If there are multiple points with the same distance, we preserve the original ordering (stable).

# In[3]:


#[TASK 2.1]
def get_k_nearest_neighbors(training_data, test_point, k):
    """
    Args:
        training_data: list of tuples [(feature_vector, label), ...]
        test_point: list of numbers (the point we're classifying)
        k: number of neighbors to consider
    Returns:
        list of labels of the k nearest neighbors
    """
    # TODO: Return the k nearest neighbors to the test_point using the euclidean_distance(vec_p, vec_q) function

    #[SOLUTION]
    distances = []
    for feature_vector, label in training_data:
        dist = euclidean_distance(feature_vector, test_point)
        distances.append((dist, label))

    # Sort by distance
    distances.sort(key=lambda x: x[0])

    # Return the labels of the k closest points
    k_neighbors = [label for _, label in distances[:k]]
    return k_neighbors
    #[/SOLUTION]
#[/TASK]

#[TESTCASES 2.1]
training_data = [
    ([1, 2], 'A'),
    ([2, 3], 'B'),
    ([3, 4], 'A'),
    ([5, 5], 'B')
]

assert get_k_nearest_neighbors(training_data, [1.5, 2.5], k=2) == ['A', 'B']
assert get_k_nearest_neighbors(training_data, [4, 4], k=1) == ['A']
assert get_k_nearest_neighbors(training_data, [0, 0], k=3) == ['A', 'B', 'A']
print('All test cases passed!') 
#[/TESTCASES]


# ## Task 3 - Prediction

# ### Task 3.1 - Compute Majority Voting
# 
# Once we have the `k` nearest neighbors, we need to decide the final label, which is the label that appears the most frequently. If there is a tie, return the label that appears first.

# In[4]:


#[TASK 3.1]
def knn_majority_vote(neighbors):
    """
    Args:
        neighbors: list of labels
    Returns:
        The label that appears most frequently.
        If there's a tie, return the label that appears first
    """
    # TODO: Return the most common label in neighbors


    #[SOLUTION]
    from collections import Counter
    count = Counter(neighbors)
    most_common_label = count.most_common(1)[0][0]
    return most_common_label
    #[/SOLUTION]
#[/TASK]

#[TESTCASES 3.1]
assert knn_majority_vote(['A', 'B', 'A']) == 'A'
assert knn_majority_vote(['B', 'B', 'A']) == 'B'
assert knn_majority_vote(['A', 'A', 'A']) == 'A'
assert knn_majority_vote(['A', 'B']) == 'A'
assert knn_majority_vote(['B', 'A']) == 'B'
print('All test cases passed!') 
#[/TESTCASES]


# ### Task 3.2 - Compute KNN regression prediction
# 
# In this, you will implement K-Nearest Neighbors (KNN) for **regression** using only Python lists and basic functions. 

# In[5]:


#[TASK 3.2]
def knn_regression(X_train, y_train, x_query, k):
    """
    Args:
        X_train (list[list[float]]): Training features
        y_train (list[float]): Target values
        x_query (list[float]): Query point
        k (int): Number of neighbors

    Returns:
        float: Predicted target value by averaging the k nearest neighbors. 
    """
    #TODO: Implement the KNN regression algorithm

    #[SOLUTION]

    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], x_query)
        distances.append((dist, y_train[i]))

    distances.sort()
    k_nearest = distances[:k]
    if not k_nearest: # Handle case where no neighbors are found (e.g., empty X_train)
        return 0.0 
    prediction = sum(val for _, val in k_nearest) / len(k_nearest) # fixed from k to len(k_nearest)
    return prediction
    #[/SOLUTION]
#[/TASK]

#[TESTCASES 3.2]
X_train = [[1], [2], [3], [4], [5]]
y_train = [1.1, 1.9, 3.0, 3.9, 5.1]
x_query = [2.5]
assert math.isclose(knn_regression(X_train, y_train, x_query, 2), 2.45, rel_tol=1e-5)

X_train = [[1], [2], [3]]
y_train = [1, 2, 3]
x_query = [2.1]
assert math.isclose(knn_regression(X_train, y_train, x_query, 1), 2, rel_tol=1e-5)

X_train = [[1], [2], [3]]
y_train = [1, 2, 3]
x_query = [2]
assert math.isclose(knn_regression(X_train, y_train, x_query, 3), 2, rel_tol=1e-5)

X_train = [[1, 2], [2, 3], [3, 4]]
y_train = [10, 20, 30]
x_query = [2, 2.5]
assert math.isclose(knn_regression(X_train, y_train, x_query, 2), 15.0, rel_tol=1e-5)

print('All test cases passed!') 

#[/TESTCASES]


# ## Task 4 - Wrapping in Classes

# ### Task 4.1 -  KNN Classifier

# Here we combine everything into a reusable class to train your own model and make predictions! 

# In[6]:


#[TASK 4.1]
class KNNClassifier:

    def __init__(self, k=3):
        self.k = k
        self.training_data = []  # Will hold tuples of (feature_vector, label)

    def fit(self, X, y):
        """
        Args:
            X : list of feature vectors
            y : list of labels corresponding to the feature vectors

        Returns:
            None
        """
        #TODO: Store the training data.

        #[SOLUTION]
        self.training_data = list(zip(X, y))
        #[/SOLUTION]


    def predict(self, X_test):
        """
        Predict the class label for each test point in X_test.

        Parameters:
            X_test : list of feature vectors to classify

        Returns:
            list of predicted labels 
        """
        #TODO: Predict the class label for each test point in X_test

        #[SOLUTION]
        predictions = []
        for test_point in X_test:
            neighbors = get_k_nearest_neighbors(self.training_data, test_point, self.k)
            predicted_label = knn_majority_vote(neighbors)
            predictions.append(predicted_label)
        return predictions
        #[/SOLUTION]

#[/TASK]

#[TESTCASES 4.1]
knn = KNNClassifier(k=3)
X_train = [[1,2],[2,3],[3,4],[5,5]]
y_train = ['A','B','A','B']
knn.fit(X_train, y_train)

X_test = [[1.5,2.5],[4,4]]
assert knn.predict(X_test) == ['A', 'B']

knn = KNNClassifier(k=1)
X_train = [[1,1],[2,2],[3,3],[4,4]]
y_train = ['A','A','B','B']
knn.fit(X_train, y_train)

X_test = [[1.5,1.5],[3.5,3.5]]
assert knn.predict(X_test) == ['A', 'B']

print('All test cases passed!') 
#[/TESTCASES]


# ### Task 4.2 - KNN Regressor
# 
# Similarly, we do the same for the regressor.

# In[7]:


#[TASK 4.2]
class KNNRegressor(KNNClassifier):
    def predict(self, X_test):
        """
        Args:
            X_test : list of feature vectors to predict

        Returns:
            list of predicted target values 
        """

        # TODO: Predict the target value for each test point in X_test

        #[SOLUTION]
        predictions = []
        # Separate training features and target values from the stored training_data
        # self.training_data is a list of (feature_vector, target_value) tuples
        X_train_fit = [item[0] for item in self.training_data]
        y_train_fit = [item[1] for item in self.training_data]

        for test_point in X_test:
            # Call the knn_regression function for each test point
            predicted_value = knn_regression(X_train_fit, y_train_fit, test_point, self.k)
            predictions.append(predicted_value)
        return predictions
        #[/SOLUTION]
#[/TASK]

#[TESTCASES 4.2]

# Test Case 1
regressor = KNNRegressor(k=3)
X_train = [[1], [2], [3], [4], [5]]
y_train = [10.0, 20.0, 30.0, 40.0, 50.0]
regressor.fit(X_train, y_train)
X_test = [[2.5], [4.5]]
predictions = regressor.predict(X_test)
assert math.isclose(predictions[0], 20.0)
assert math.isclose(predictions[1], 40.0)

# Test Case 2
regressor = KNNRegressor(k=1)
X_train = [[1, 1], [2, 2], [3, 3]]
y_train = [10.0, 20.0, 30.0]
regressor.fit(X_train, y_train)
X_test = [[1.2, 1.2], [2.8, 2.8], [0.5, 0.5]]
predictions = regressor.predict(X_test)
assert math.isclose(predictions[0], 10.0)
assert math.isclose(predictions[1], 30.0)
assert math.isclose(predictions[2], 10.0)

print('All test cases passed!') 
#[/TESTCASES]        


# ## Task Bonus - Visualizing the Effect of `k` in KNN
# 
# The code below visualizes how k (number of neighbors) impacts KNN classification accuracy on the provided synthetic dataset. Determine the optimal k based on the results and answer the corresponding coursemology question.

# In[8]:


# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.datasets import make_moons
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score

# # Step 1: Generate synthetic data
# X, y = make_moons(n_samples=300, noise=0.3, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Step 2: Try multiple values of k and store accuracy
# k_values = list(range(1, 16))
# train_accuracies = []
# test_accuracies = []

# for k in k_values:
#     clf = KNeighborsClassifier(n_neighbors=k)
#     clf.fit(X_train, y_train)

#     # Accuracy on training data
#     y_train_pred = clf.predict(X_train)
#     train_acc = accuracy_score(y_train, y_train_pred)
#     train_accuracies.append(train_acc)

#     # Accuracy on test data
#     y_test_pred = clf.predict(X_test)
#     test_acc = accuracy_score(y_test, y_test_pred)
#     test_accuracies.append(test_acc)

# # Step 3: Plot accuracy vs k for both train and test
# plt.figure(figsize=(8, 5))
# plt.plot(k_values, train_accuracies, marker='o', linestyle='-', label='Train Accuracy')
# plt.plot(k_values, test_accuracies, marker='s', linestyle='--', label='Test Accuracy')
# plt.title('KNN Accuracy vs k')
# plt.xlabel('k (Number of Neighbors)')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Step 4: Plot decision boundaries for selected values of k
# def plot_decision_boundary(clf, X, y, title):
#     x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
#     y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
#     xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
#                          np.linspace(y_min, y_max, 300))

#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)

#     plt.figure(figsize=(6, 5))
#     plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
#     plt.title(title)
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.grid(True)
#     plt.show()

# # Show decision boundaries for k = 1, 5, 15
# for k in [1, 5, 15]:
#     clf = KNeighborsClassifier(n_neighbors=k)
#     clf.fit(X_train, y_train)
#     plot_decision_boundary(clf, X, y, title=f"KNN Decision Boundary (k={k})")


# # Task 5 - Practical
# 
# Train a KNN classifier on the training dataset using scikit-learn. Leverage everything you learnt about KNNs such as feature transformations and tune hyperparameters to optimize performance.
# 
# You will get full marks if your accuracy reaches 50%. You may not use or access X_test and y_test in your code, as this defeats the purpose of a hidden test set

# In[ ]:


from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

# Load dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw_people.data  # Flattened images
y = lfw_people.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

#[TASK 5.1]
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import numpy as np

def train_model(X_train, y_train):
    """
    Args:
        X_train: Training feature vectors
        y_train: Training labels
    Returns:
        A trained sklearn model, your model will be used to predict the labels of test data
    """
    # TODO: Train and return a kNN classifier
    # If using PCA, use a pipeline to combine PCA and kNN
    # ie. from sklearn.pipeline import make_pipeline
    # When .predict is called, the model should be able to perform any necessary 
    # transformations (like PCA) on the test data automatically

    #[SOLUTION]
    model = make_pipeline(
        PCA(n_components=150, whiten=True, random_state=42),
        KNeighborsClassifier(n_neighbors=3)
    )
    model.fit(X_train, y_train)
    return model
    #[/SOLUTION]
#[/TASK]

#[TESTCASES 5.1]
# Our test cases will use your trained sklearn model to predict the labels of the test data.
# Our test does will only display the model's accuracy score.
# You will automatically receive points if your model achieves at least 50% accuracy on the test set.
# Manual review will be conducted for additional feedback.
# Note: If your model is poorly designed or performs poorly, points may be deducted.
model = train_model(X_train, y_train)
# Check if the model can predict
predictions = model.predict(X_test)
assert len(predictions) == len(X_test)
accuracy_score = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy_score:.2f}")
#[/TESTCASES]     

