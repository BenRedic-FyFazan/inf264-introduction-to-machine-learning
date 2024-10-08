import numpy as np
from collections import Counter

def majority_class(y):
    """
    Determines the majority class in a list of labels.
    """
    return Counter(y).most_common(1)[0][0]


def valid_criterion(criterion):
    """
    Validates the splitting criterion.
    """
    valid_criteria = {'entropy', 'gini'}
    if criterion not in valid_criteria:
        raise ValueError('Invalid criterion. Supported criteria are {}'.format(valid_criteria))
    return criterion


def valid_max_feature(max_feature):
    """
    Validates the maximum number of features to consider when looking for the best split.
    """
    valid_max_features = {'sqrt', 'log2', None}
    if max_feature not in valid_max_features:
        raise ValueError('Invalid max_features. Supported max_features are {}'.format(valid_max_features))
    return max_feature

def valid_split_method(split_method):
    """
    Validates the method used to determine split thresholds.
    """
    split_methods = {'mean', 'median', 'unique'}
    if split_method not in split_methods:
        raise ValueError('Invalid split_method. Supported methods are {}'.format(split_methods))
    return split_method


class Node:
    """
    Represents an internal node in the decision tree.
    """
    def __init__(self, feature, threshold, left, right):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


class Leaf:
    """
    Represents a leaf node in the decision tree.
    """
    def __init__(self, label):
        self.label = label


class DecisionTree:
    """
    A decision tree classifier.

    Parameters
    ----------
    criterion : str, default='entropy'
        The function to measure the quality of a split. Supported criteria are 'entropy' and 'gini'.
    max_depth : int or None, default=None
        The maximum depth of the tree. If None, nodes are expanded until all leaves are pure.
    max_features : str or None, default=None
        The number of features to consider when looking for the best split.
        Supported options are 'sqrt', 'log2', or None (all features).
    split_method : str, default='mean'
        Method to determine split thresholds. Must be 'mean', 'median', or 'unique'.
    random_state : int or None, default=None
        Seed used by the random number generator.

    Attributes
    ----------
    tree : Node or Leaf
        The root node of the trained decision tree.
    num_classes : int
        Number of unique classes in the target labels.
    random_state : np.random.RandomState
        Random number generator instance.

    Notes
    ---------
    The Parameters are also set as attributes.
    """
    def __init__(self, criterion='entropy', max_depth=None, max_features=None, split_method='mean', random_state=None):
        self.criterion = valid_criterion(criterion)
        self.max_features = valid_max_feature(max_features)
        self.split_method = valid_split_method(split_method)
        self.max_depth = max_depth
        self.num_classes = None
        self.tree = None

        # Allow seeded randoms
        if random_state is not None:
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = np.random.RandomState

    def fit(self, feature_matrix, labels):
        """
        Builds the decision tree from the training data.
        """
        self.num_classes = len(set(labels))
        self.tree = self._build_tree(feature_matrix, labels, depth=0)


    def predict(self, x):
        """
        Predicts class labels for the given samples.
        """
        predictions = [self._traverse_tree(sample, self.tree) for sample in x]
        return np.array(predictions)


    def _build_tree(self, feature_matrix, labels, depth):
        """
        Recursively builds the decision tree.
        """
        # Check if all labels are the same.
        if len(set(labels)) == 1:
            # Set returns a set containing all unique members of the input collection
            # IE if len == 1 then all labels are equal.
            return Leaf(labels[0])

        # Check if all features are identical
        if np.all(np.var(feature_matrix, axis=0) == 0):
            return Leaf(majority_class(labels))

        # Check if max_depth is reached
        if self.max_depth is not None and depth >= self.max_depth:
            return Leaf(majority_class(labels))

        # Find the best feature and threshold to split on
        best_feature, best_threshold = self._find_best_split(feature_matrix, labels)

        # If no valid split is found
        if best_feature is None:
            return Leaf(majority_class(labels))

        # Split the dataset
        left_indices = feature_matrix[:, best_feature] <= best_threshold
        right_indices = feature_matrix[:, best_feature] > best_threshold
        feature_matrix_left, labels_left = feature_matrix[left_indices], labels[left_indices]
        feature_matrix_right, labels_right = feature_matrix[right_indices], labels[right_indices]

        # Recursively build the left and right subtrees
        left_subtree = self._build_tree(feature_matrix_left, labels_left, depth + 1)
        right_subtree = self._build_tree(feature_matrix_right, labels_right, depth + 1)

        # Return the decision node
        return Node(best_feature, best_threshold, left_subtree, right_subtree)


    def _find_best_split(self, feature_matrix, labels):
        """
        Finds the best feature and threshold to split the data.
        """
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None

        for feature in self._features_to_consider(feature_matrix):
            feature_values = feature_matrix[:, feature] # Slice away the unconsidered features from the array

            # Calculate split thresholds.
            if self.split_method == 'unique':
                thresholds = np.unique(feature_values)
            elif self.split_method == 'median':
                thresholds = [np.median(feature_values)]
            else:
                thresholds = [np.mean(feature_values)]

            for threshold in thresholds:
                gain = self._information_gain(labels, feature_values, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold


    def _features_to_consider(self, feature_matrix):
        """
        Determines which features to consider for splitting based on `max_features`.
        """
        if self.max_features == 'sqrt':
            num_features_to_consider = max(1, int(np.floor(np.sqrt(feature_matrix.shape[1]))))
            return self.random_state.choice(feature_matrix.shape[1], num_features_to_consider, replace=False)

        elif self.max_features == 'log2':
            num_features_to_consider = max(1, int(np.floor(np.log2(feature_matrix.shape[1]))))
            return self.random_state.choice(feature_matrix.shape[1], num_features_to_consider, replace=False)

        else:
            # max_features is None
            return np.arange(feature_matrix.shape[1])


    def _information_gain(self, y, feature_values, threshold):
        """
        Calculates the information gain of a potential split.

        Information gain equation:
            IG(x) = H(y) - H(y|x)
        Where:
            H(y) is the impurity before the split.
            H(y|x) is the weighted impurity after splitting on feature x.

        Notes
        -----
        Returns `-np.inf` if the split does not divide the data.
        """
        # Impurity before splitting -> H(y)
        parent_impurity = self._impurity(y)

        # Split the data
        y_left = y[feature_values <= threshold]
        y_right = y[feature_values > threshold]

        # Avoid division/splitting by zero
        if len(y_left) == 0 or len(y_right) == 0:
            return -np.inf

        # Weighted impurity of children after splitting -> H(y|x)
        impurity_left = self._impurity(y_left)
        impurity_right = self._impurity(y_right)
        weighted_impurity_left = ( len(y_left) / len(y) ) * impurity_left
        weighted_impurity_right = ( len(y_right) / len(y) ) * impurity_right
        weighted_impurity = weighted_impurity_left + weighted_impurity_right

        # Information gain -> IG(x) = H(y) - H(y|x)
        gain = parent_impurity - weighted_impurity
        return gain

    def _impurity(self, y):
        """
        Calculates the impurity of a potential split.
        """
        probabilities = np.bincount(y) / len(y)

        if self.criterion == 'gini':
            """
            Gini formula as given by lecture: 
                G(x) =  sum[ p(x==x_i) * (1 - p(x == x_i))]
            Simplify: 
                p = p(x==x_i)
                G(x) = sum[ p * (1-p)] 
                G(x) = sum[1p - p^2]
                G(x) = sum[p] - sum[p^2]
            We know that the sum of probabilities = 1, ie sum[y] = 1
                G(x) = 1 - sum[p^2]
            """
            return 1.0 - np.sum(probabilities ** 2)

        elif self.criterion == 'entropy':
            """
            Entropy formula as given by lecture:
                H(x) = -sum[ p(x==x_i) * log_2(p(x==x_i)) ]
            Simplify:
                p = p(x==x_i)
                H(x) = 0 - sum[p * log_2(p)]
            0.000000001 is added to each probability to prevent log_2(0).
            """
            return -np.sum(probabilities * np.log2(probabilities + 1e-9))

        else:
            raise ValueError(f"Unknown criterion {self.criterion}")


    def _traverse_tree(self, x, node):
        """
        Traverses the decision tree to make a prediction for a single sample.
        """
        if isinstance(node, Leaf):
            return node.label
        else:
            if x[node.feature] <= node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)


class RandomForest:
    """
    A random forest classifier consisting of multiple decision trees.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    max_depth : int or None, default=None
        The maximum depth of each tree. If None, nodes are expanded until all leaves are pure.
    criterion : str, default='entropy'
        The function to measure the quality of a split. Supported criteria are 'entropy' and 'gini'.
    max_features : str or None, default='sqrt'
        The number of features to consider when looking for the best split.
        Supported options are 'sqrt', 'log2', or None (all features).
    random_state : int or None, default=None
        Seed used by the random number generator.

    Attributes
    ----------
    forest : list of DecisionTree
        The collection of trained decision trees.
    random_state : np.random.RandomState
        Random number generator instance.

    Notes
    ---------
    The Parameters are also set as attributes.
    """
    def __init__(self, n_estimators=100, max_depth=None, criterion='entropy', max_features='sqrt', random_state=None):
        self.criterion = valid_criterion(criterion)
        self.max_features = valid_max_feature(max_features)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.forest = []

        # Allow seeded randoms
        if random_state is not None:
            self.random_seed = random_state
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_seed = np.random
            self.random_state = np.random.RandomState(self.random_seed)


    def fit(self, feature_matrix, labels):
        """
        Builds the random forest from the training data.
        """
        num_samples = feature_matrix.shape[0]

        for _ in range(self.n_estimators):
            # Create a random subset of the data with replacement
            bootstrap_indices = self.random_state.choice(num_samples, num_samples, replace=True)
            bootstrap_feature_matrix = feature_matrix[bootstrap_indices]
            bootstrap_labels = labels[bootstrap_indices]

            # Create and train a decision tree on the bootstrapped data
            tree = DecisionTree(
                max_depth=self.max_depth,
                criterion=self.criterion,
                max_features=self.max_features,
                random_state=self.random_seed
            )
            tree.fit(bootstrap_feature_matrix, bootstrap_labels)
            self.forest.append(tree)

    def predict(self, feature_matrix):
        """
        Predicts class labels for the given samples using majority voting from all trees.
        """
        predictions = []
        majority_votes = []

        # Collect predictions from all trees
        for tree in self.forest:
            predictions.append(tree.predict(feature_matrix))

        # Perform majority voting
        for sample in np.array(predictions).T:
            majority_votes.append(majority_class(sample))

        return np.array(majority_votes)

