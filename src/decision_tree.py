import numpy as np
import pandas as pd
from collections import Counter
import math

class DecisionTreeNode:
    """
    A single node in the decision tree structure
    Each node represents either a decision point or a leaf with a prediction
    """
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None, samples=0, impurity=0.0):
        # For internal nodes (decision points)
        self.feature_index = feature_index    # Which feature to split on
        self.threshold = threshold            # The value to split the feature at
        self.left = left                      # Left child node (feature <= threshold)
        self.right = right                    # Right child node (feature > threshold)
        
        # For leaf nodes (final predictions)
        self.value = value                    # The predicted class for this leaf
        
        # Additional information for feature importance calculation
        self.samples = samples                # Number of samples in this node
        self.impurity = impurity             # Impurity of this node

class DecisionTree:
    """
    Complete Decision Tree implementation from scratch for heart disease classification
    Uses information theory (Gini or Entropy) to find the best splits
    Enhanced version with proper feature importance calculation
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, criterion='gini'):
        """
        Initialize the Decision Tree with hyperparameters
        
        Parameters:
        -----------
        max_depth : int
            Maximum depth of the tree to prevent overfitting
        min_samples_split : int
            Minimum number of samples required to split an internal node
        min_samples_leaf : int
            Minimum number of samples required to be at a leaf node
        criterion : str
            The function to measure the quality of a split ('gini' or 'entropy')
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.root = None                      # Root node of the tree
        self.feature_names = None             # Names of features for interpretation
        self.n_features = None                # Number of features
        self.feature_importances_ = None      # Feature importance scores
    
    def _calculate_gini(self, y):
        """
        Calculate Gini impurity for a set of labels
        Gini = 1 - Σ(p_i^2) where p_i is the probability of class i
        
        Parameters:
        -----------
        y : array-like
            Array of class labels
            
        Returns:
        --------
        float
            Gini impurity value (0 = pure, higher = more mixed)
        """
        if len(y) == 0:
            return 0
        
        # Count occurrences of each class
        _, counts = np.unique(y, return_counts=True)
        
        # Calculate probabilities for each class
        probabilities = counts / len(y)
        
        # Calculate Gini impurity: 1 - sum of squared probabilities
        gini = 1 - np.sum(probabilities ** 2)
        return gini
    
    def _calculate_entropy(self, y):
        """
        Calculate entropy for a set of labels
        Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i
        
        Parameters:
        -----------
        y : array-like
            Array of class labels
            
        Returns:
        --------
        float
            Entropy value (0 = pure, higher = more mixed)
        """
        if len(y) == 0:
            return 0
        
        # Count occurrences of each class
        _, counts = np.unique(y, return_counts=True)
        
        # Calculate probabilities for each class
        probabilities = counts / len(y)
        
        # Calculate entropy: -sum of p*log2(p)
        # Add small epsilon to avoid log(0)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
        return entropy
    
    def _calculate_impurity(self, y):
        """
        Calculate impurity using the specified criterion
        
        Parameters:
        -----------
        y : array-like
            Array of class labels
            
        Returns:
        --------
        float
            Impurity value based on chosen criterion
        """
        if self.criterion == 'gini':
            return self._calculate_gini(y)
        elif self.criterion == 'entropy':
            return self._calculate_entropy(y)
        else:
            raise ValueError("Criterion must be 'gini' or 'entropy'")
    
    def _calculate_information_gain(self, parent, left_child, right_child):
        """
        Calculate information gain from a potential split
        Information Gain = Parent Impurity - Weighted Average of Children Impurities
        
        Parameters:
        -----------
        parent : array-like
            Labels in the parent node
        left_child : array-like
            Labels that would go to left child
        right_child : array-like
            Labels that would go to right child
            
        Returns:
        --------
        float
            Information gain from this split (higher is better)
        """
        # Calculate sizes for weighting
        n_parent = len(parent)
        n_left = len(left_child)
        n_right = len(right_child)
        
        # If split results in empty child, no information gain
        if n_left == 0 or n_right == 0:
            return 0
        
        # Calculate impurities
        parent_impurity = self._calculate_impurity(parent)
        left_impurity = self._calculate_impurity(left_child)
        right_impurity = self._calculate_impurity(right_child)
        
        # Calculate weighted average of child impurities
        left_weight = n_left / n_parent
        right_weight = n_right / n_parent
        weighted_child_impurity = (left_weight * left_impurity + 
                                 right_weight * right_impurity)
        
        # Information gain is the reduction in impurity
        information_gain = parent_impurity - weighted_child_impurity
        return information_gain
    
    def _find_best_split(self, X, y):
        """
        Find the best feature and threshold to split the data
        Tests all possible splits and returns the one with highest information gain
        Enhanced version with sorted values for better performance
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        y : array-like, shape (n_samples,)
            Target labels
            
        Returns:
        --------
        tuple
            (best_feature_index, best_threshold, best_gain)
        """
        best_gain = -1
        best_feature_index = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        # Try every feature as a potential split
        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            
            # Sort the feature values along with corresponding labels
            sorted_indices = np.argsort(feature_values)
            sorted_values = feature_values[sorted_indices]
            sorted_labels = y[sorted_indices]
            
            # Try thresholds between consecutive unique values
            for i in range(len(sorted_values) - 1):
                # Skip if values are the same
                if sorted_values[i] == sorted_values[i + 1]:
                    continue
                
                # Use midpoint as threshold
                threshold = (sorted_values[i] + sorted_values[i + 1]) / 2
                
                # Split data based on this threshold
                left_y = sorted_labels[:i + 1]
                right_y = sorted_labels[i + 1:]
                
                # Skip if split results in empty children or violates min_samples_leaf
                if (len(left_y) < self.min_samples_leaf or 
                    len(right_y) < self.min_samples_leaf):
                    continue
                
                # Calculate information gain for this split
                gain = self._calculate_information_gain(sorted_labels, left_y, right_y)
                
                # Update best split if this is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold
        
        return best_feature_index, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree
        This is the core algorithm that creates the tree structure
        Enhanced version with proper node information for feature importance
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix for current node
        y : array-like, shape (n_samples,)
            Target labels for current node
        depth : int
            Current depth in the tree
            
        Returns:
        --------
        DecisionTreeNode
            Root node of the (sub)tree
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Calculate current impurity
        current_impurity = self._calculate_impurity(y)
        
        # Stopping criteria - create leaf node if any condition is met
        if (depth >= self.max_depth or                    # Maximum depth reached
            n_samples < self.min_samples_split or         # Too few samples to split
            n_classes == 1 or                             # All samples same class
            current_impurity == 0):                       # Pure node
            
            # Create leaf node with most common class
            most_common_class = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=most_common_class, samples=n_samples, impurity=current_impurity)
        
        # Find the best way to split current data
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        # If no good split found, create leaf node
        if best_feature is None or best_gain <= 0:
            most_common_class = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=most_common_class, samples=n_samples, impurity=current_impurity)
        
        # Split the data based on best split found
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        # Check minimum samples per leaf constraint (double check)
        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            most_common_class = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=most_common_class, samples=n_samples, impurity=current_impurity)
        
        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)
        
        # Create internal node with the best split
        return DecisionTreeNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
            samples=n_samples,
            impurity=current_impurity
        )
    
    def fit(self, X, y, feature_names=None):
        """
        Train the decision tree on the given data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training feature matrix
        y : array-like, shape (n_samples,)
            Training target labels
        feature_names : list, optional
            Names of the features for better interpretation
            
        Returns:
        --------
        self
            Returns the trained tree object
        """
        # Handle pandas DataFrames
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = feature_names
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Store number of features
        self.n_features = X.shape[1]
        
        print(f"Training Decision Tree for Heart Disease Classification...")
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Classes: {len(np.unique(y))} unique classes")
        print(f"Hyperparameters: max_depth={self.max_depth}, "
              f"min_samples_split={self.min_samples_split}, "
              f"criterion={self.criterion}")
        
        # Build the tree starting from root
        self.root = self._build_tree(X, y)
        
        # Calculate feature importances
        self._calculate_feature_importances(X, y)
        
        print("✅ Decision Tree training completed!")
        return self
    
    def _calculate_feature_importances(self, X, y):
        """
        Calculate feature importance based on impurity decrease
        Features that provide more impurity reduction are more important
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training feature matrix
        y : array-like, shape (n_samples,)
            Training target labels
        """
        # Initialize importance scores
        importances = np.zeros(self.n_features)
        
        def calculate_importance_recursive(node, n_node_samples):
            """Recursively calculate importance for each feature used in tree"""
            if node.value is not None:  # Leaf node
                return
            
            # Calculate importance for this split
            feature_idx = node.feature_index
            
            # Get left and right child information
            n_left = node.left.samples
            n_right = node.right.samples
            
            # Calculate weighted impurity decrease
            importance = (n_node_samples / X.shape[0]) * (
                node.impurity - 
                (n_left / n_node_samples) * node.left.impurity -
                (n_right / n_node_samples) * node.right.impurity
            )
            
            # Add to feature importance
            importances[feature_idx] += importance
            
            # Recursively calculate for child nodes
            calculate_importance_recursive(node.left, n_left)
            calculate_importance_recursive(node.right, n_right)
        
        # Calculate importance starting from root
        if self.root is not None:
            calculate_importance_recursive(self.root, self.root.samples)
        
        # Normalize importances to sum to 1
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        
        self.feature_importances_ = importances
    
    def _predict_sample(self, x, node):
        """
        Predict the class for a single sample by traversing the tree
        
        Parameters:
        -----------
        x : array-like
            Single sample features
        node : DecisionTreeNode
            Current node in the tree
            
        Returns:
        --------
        predicted class
        """
        # If we reach a leaf node, return its value
        if node.value is not None:
            return node.value
        
        # Otherwise, decide which branch to follow based on feature value
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        """
        Make predictions for new data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix to make predictions for
            
        Returns:
        --------
        array-like, shape (n_samples,)
            Predicted class labels
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        predictions = []
        
        # Predict each sample individually
        for x in X:
            prediction = self._predict_sample(x, self.root)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def feature_importance(self, X=None, y=None):
        """
        Get feature importance scores
        
        Parameters:
        -----------
        X : array-like, optional (for compatibility)
            Not used, kept for interface compatibility
        y : array-like, optional (for compatibility)
            Not used, kept for interface compatibility
            
        Returns:
        --------
        array-like
            Feature importance scores (normalized to sum to 1)
        """
        return self.feature_importances_
    
    def _get_tree_rules(self, node, rules, depth=0, condition=""):
        """
        Extract human-readable rules from the trained tree
        
        Parameters:
        -----------
        node : DecisionTreeNode
            Current node to extract rules from
        rules : list
            List to store extracted rules
        depth : int
            Current depth for indentation
        condition : str
            Current condition path
        """
        if node.value is not None:
            # Leaf node - add the final rule
            diagnosis = "Heart Disease" if node.value == 1 else "No Heart Disease"
            rules.append(f"{'  ' * depth}IF {condition} THEN {diagnosis}")
            return
        
        # Internal node - add conditions for left and right branches
        feature_name = f"feature_{node.feature_index}"
        if self.feature_names:
            feature_name = self.feature_names[node.feature_index]
        
        # Left branch condition (<=)
        left_condition = f"{condition} AND {feature_name} <= {node.threshold:.2f}" if condition else f"{feature_name} <= {node.threshold:.2f}"
        
        # Right branch condition (>)
        right_condition = f"{condition} AND {feature_name} > {node.threshold:.2f}" if condition else f"{feature_name} > {node.threshold:.2f}"
        
        # Recursively get rules for both branches
        self._get_tree_rules(node.left, rules, depth + 1, left_condition)
        self._get_tree_rules(node.right, rules, depth + 1, right_condition)
    
    def get_rules(self):
        """
        Get human-readable rules from the trained tree
        
        Returns:
        --------
        list
            List of if-then rules extracted from the tree
        """
        rules = []
        if self.root:
            self._get_tree_rules(self.root, rules)
        return rules
    
    def predict_proba(self, X):
        """
        Predict class probabilities for the given samples
        For decision trees, this returns the proportion of samples of each class
        in the leaf that each sample falls into
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        array-like, shape (n_samples, n_classes)
            Probability of each class for each sample
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        def get_leaf_probabilities(x, node):
            """Get probability distribution from leaf node"""
            if node.value is not None:
                # Leaf node - return hard prediction as probability
                if node.value == 0:
                    return [1.0, 0.0]  # No disease
                else:
                    return [0.0, 1.0]  # Disease
            
            # Internal node - continue traversing
            if x[node.feature_index] <= node.threshold:
                return get_leaf_probabilities(x, node.left)
            else:
                return get_leaf_probabilities(x, node.right)
        
        probabilities = []
        for x in X:
            prob = get_leaf_probabilities(x, self.root)
            probabilities.append(prob)
        
        return np.array(probabilities)

# Example usage and testing
if __name__ == "__main__":
    # Test the decision tree with sample heart disease data
    
    def create_sample_heart_data(n_samples=200, random_state=42):
        """Create sample heart disease data for testing"""
        np.random.seed(random_state)
        
        # Generate realistic heart disease features
        data = {}
        
        # Age (normally distributed around 54)
        data['age'] = np.clip(np.random.normal(54, 9, n_samples), 29, 77)
        
        # Sex (68% male)
        data['sex'] = np.random.choice([0, 1], n_samples, p=[0.32, 0.68])
        
        # Chest pain type
        data['cp'] = np.random.choice([0, 1, 2, 3], n_samples)
        
        # Blood pressure
        data['trestbps'] = np.clip(np.random.normal(131, 17, n_samples), 94, 200)
        
        # Cholesterol
        data['chol'] = np.clip(np.random.normal(246, 51, n_samples), 126, 564)
        
        # Create realistic target based on risk factors
        risk_score = (
            (data['age'] - 40) * 0.02 +
            data['sex'] * 0.3 +
            (data['trestbps'] - 120) * 0.01 +
            (data['chol'] - 200) * 0.001
        )
        risk_score += np.random.normal(0, 0.3, n_samples)
        
        threshold = np.percentile(risk_score, 50)
        target = (risk_score > threshold).astype(int)
        
        # Create feature matrix
        X = np.column_stack([data['age'], data['sex'], data['cp'], 
                           data['trestbps'], data['chol']])
        
        return X, target
    
    def simple_train_test_split(X, y, test_size=0.2, random_state=42):
        """Simple train-test split function"""
        np.random.seed(random_state)
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
    
    def simple_accuracy_score(y_true, y_pred):
        """Simple accuracy calculation"""
        return np.mean(y_true == y_pred)
    
    print("Testing Decision Tree Implementation for Heart Disease")
    print("=" * 60)
    
    # Create sample dataset
    X, y = create_sample_heart_data(n_samples=200, random_state=42)
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol']
    
    print(f"Dataset created: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Disease prevalence: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = simple_train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the tree
    dt = DecisionTree(max_depth=6, min_samples_split=5, criterion='gini')
    dt.fit(X_train, y_train, feature_names=feature_names)
    
    # Make predictions
    y_pred = dt.predict(X_test)
    
    # Evaluate performance
    accuracy = simple_accuracy_score(y_test, y_pred)
    print(f"\nDecision Tree Accuracy: {accuracy:.4f}")
    
    # Show class distribution
    unique_true, counts_true = np.unique(y_test, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    
    print(f"\nActual class distribution:")
    for cls, count in zip(unique_true, counts_true):
        label = "No Disease" if cls == 0 else "Disease"
        print(f"  {label}: {count}")
    
    print(f"\nPredicted class distribution:")
    for cls, count in zip(unique_pred, counts_pred):
        label = "No Disease" if cls == 0 else "Disease"
        print(f"  {label}: {count}")
    
    # Feature importance
    importances = dt.feature_importance()
    print(f"\nFeature Importances:")
    for i, (name, importance) in enumerate(zip(feature_names, importances)):
        print(f"{name:12} {importance:.4f}")
    
    # Sample rules (first 5)
    rules = dt.get_rules()
    print(f"\nNumber of decision rules: {len(rules)}")
    print("First 5 rules:")
    for rule in rules[:5]:
        print(rule)
    
    # Test probabilities
    probas = dt.predict_proba(X_test[:3])
    print(f"\nPrediction probabilities for first 3 test samples:")
    for i, (prob, true_val, pred_val) in enumerate(zip(probas, y_test[:3], y_pred[:3])):
        print(f"Sample {i+1}: True={true_val}, Predicted={pred_val}")
        print(f"  P(No Disease): {prob[0]:.3f}, P(Disease): {prob[1]:.3f}")
    
    print("\n✅ Decision Tree test completed successfully!")