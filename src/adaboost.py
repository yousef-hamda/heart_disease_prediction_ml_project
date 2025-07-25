import numpy as np
import pandas as pd
from collections import Counter
import math
import warnings
warnings.filterwarnings('ignore')

class DecisionStump:
    """
    Simple decision tree with depth 1 - basically just one split
    This is called a "stump" because it's like a tree that has been cut down
    Enhanced version with better error handling for heart disease prediction
    """
    
    def __init__(self):
        self.feature_idx = None      # which column to split on
        self.threshold = None        # the value to split at
        self.polarity = 1           # 1 or -1, determines left vs right direction
        self.alpha = None           # how much weight this stump gets in final vote
    
    def predict(self, X):
        """
        Make predictions - straightforward implementation
        Enhanced with better array handling
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        num_samples = X.shape[0]
        predictions = np.ones(num_samples)  # start with all 1s
        
        # Apply the split rule
        if self.polarity == 1:
            predictions[X[:, self.feature_idx] < self.threshold] = -1
        else:
            predictions[X[:, self.feature_idx] >= self.threshold] = -1
        
        return predictions

class AdaBoost:
    """
    AdaBoost implementation from scratch for heart disease prediction
    Enhanced version with better numerical stability and error handling
    Based on Freund & Schapire 1997 with improvements for robustness
    """
    
    def __init__(self, num_estimators=50, learning_rate=1.0, random_state=None):
        """
        Set up the AdaBoost classifier for heart disease prediction
        
        Args:
            num_estimators: how many weak learners to train
            learning_rate: shrinks each classifier's contribution 
            random_state: for reproducible results
        """
        self.num_estimators = num_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # These get filled during training
        self.stumps = []              # list of decision stumps
        self.stump_weights = []       # alpha values for each stump
        self.training_errors = []     # keep track of errors for analysis
        self.feature_names = None
        self.n_features = None
        self.feature_importances_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _convert_labels_to_binary(self, y):
        """
        AdaBoost needs labels to be +1 and -1, not 0 and 1
        Enhanced version with better label mapping for heart disease
        """
        unique_labels = np.unique(y)
        
        if len(unique_labels) == 2:
            # Binary classification - map to -1 and +1
            # For heart disease: 0 (no disease) -> -1, 1 (disease) -> +1
            sorted_labels = np.sort(unique_labels)
            label_mapping = {sorted_labels[0]: -1, sorted_labels[1]: 1}
            binary_labels = np.array([label_mapping[label] for label in y])
            return binary_labels, label_mapping
        else:
            # Multi-class - use one vs all approach
            # Take most common class as +1, rest as -1
            most_common = Counter(y).most_common(1)[0][0]
            binary_labels = np.where(y == most_common, 1, -1)
            label_mapping = {-1: "others", 1: most_common}
            return binary_labels, label_mapping
    
    def _find_best_stump(self, X, y, sample_weights):
        """
        Find the decision stump that minimizes weighted error
        This is the core of AdaBoost - finding weak learners
        Enhanced version with better numerical stability
        """
        num_samples, num_features = X.shape
        best_stump = DecisionStump()
        min_error = float('inf')  # start with worst possible error
        
        # Try every feature
        for feature_idx in range(num_features):
            feature_values = X[:, feature_idx]
            
            # Get sorted unique values for better threshold selection
            sorted_indices = np.argsort(feature_values)
            sorted_values = feature_values[sorted_indices]
            sorted_labels = y[sorted_indices]
            sorted_weights = sample_weights[sorted_indices]
            
            # Try thresholds between consecutive different values
            unique_values = []
            for i in range(len(sorted_values)):
                if i == 0 or sorted_values[i] != sorted_values[i-1]:
                    unique_values.append(sorted_values[i])
            
            # Try each unique value as threshold
            for threshold in unique_values:
                # Try both polarities (< vs >=)
                for polarity in [1, -1]:
                    # Create predictions for this stump
                    predictions = np.ones(num_samples)
                    if polarity == 1:
                        predictions[feature_values < threshold] = -1
                    else:
                        predictions[feature_values >= threshold] = -1
                    
                    # Calculate weighted error
                    # This is key: we weight errors by sample importance
                    wrong_predictions = predictions != y
                    weighted_error = np.sum(sample_weights[wrong_predictions])
                    
                    # Add small epsilon to avoid exact zero error
                    weighted_error = max(weighted_error, 1e-10)
                    
                    # Keep track of best stump so far
                    if weighted_error < min_error:
                        min_error = weighted_error
                        best_stump.feature_idx = feature_idx
                        best_stump.threshold = threshold
                        best_stump.polarity = polarity
        
        return best_stump, min_error
    
    def fit(self, X, y):
        """
        Train the AdaBoost classifier for heart disease prediction
        Enhanced version with better error handling and numerical stability
        """
        # Handle pandas input
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Store number of features
        self.n_features = X.shape[1]
        
        # Convert to binary labels
        binary_y, self.label_mapping = self._convert_labels_to_binary(y)
        
        num_samples = X.shape[0]
        
        # Initialize sample weights - everyone starts equal
        sample_weights = np.full(num_samples, 1.0 / num_samples)
        
        # Reset everything for new training
        self.stumps = []
        self.stump_weights = []
        self.training_errors = []
        
        print("Starting AdaBoost training for heart disease classification...")
        print(f"Dataset: {num_samples} patients, {X.shape[1]} medical features")
        
        # Main AdaBoost loop
        for iteration in range(self.num_estimators):
            # Step 1: Find best weak learner for current weights
            best_stump, min_error = self._find_best_stump(X, binary_y, sample_weights)
            
            # Step 2: Check if we should stop
            # If error >= 0.5, the classifier is worse than random
            if min_error >= 0.5:
                if iteration == 0:
                    # If first classifier is bad, use it anyway but with tiny weight
                    min_error = 0.5 - 1e-10
                else:
                    print(f"Stopping at iteration {iteration}: error {min_error:.4f} >= 0.5")
                    break
            
            # Ensure minimum error to avoid numerical issues
            min_error = max(min_error, 1e-10)
            min_error = min(min_error, 1 - 1e-10)
            
            # Step 3: Calculate classifier weight (alpha)
            # This is the famous AdaBoost formula!
            alpha = self.learning_rate * 0.5 * np.log((1 - min_error) / min_error)
            best_stump.alpha = alpha
            
            # Step 4: Get predictions from this stump
            stump_predictions = best_stump.predict(X)
            
            # Step 5: Update sample weights
            # Increase weight of misclassified samples, decrease weight of correct ones
            # This makes the next classifier focus on the mistakes
            weight_multiplier = np.exp(-alpha * binary_y * stump_predictions)
            sample_weights *= weight_multiplier
            
            # Step 6: Normalize weights so they sum to 1
            weight_sum = np.sum(sample_weights)
            if weight_sum > 0:
                sample_weights /= weight_sum
            else:
                # Fallback to uniform weights if all weights become zero
                sample_weights = np.full(num_samples, 1.0 / num_samples)
            
            # Save this classifier and its stats
            self.stumps.append(best_stump)
            self.stump_weights.append(alpha)
            self.training_errors.append(min_error)
            
            # Print progress every 10 iterations
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.num_estimators}: "
                      f"error = {min_error:.4f}, α = {alpha:.4f}")
        
        # Calculate feature importances
        self._calculate_feature_importances()
        
        print(f"Training complete! Created {len(self.stumps)} weak learners.")
        return self
    
    def _calculate_feature_importances(self):
        """
        Calculate which features are used most often, weighted by stump importance
        Enhanced version with proper normalization for heart disease features
        """
        if not self.stumps or self.n_features is None:
            self.feature_importances_ = None
            return
        
        importance_scores = np.zeros(self.n_features)
        
        # Weight by stump importance (alpha values)
        for stump, weight in zip(self.stumps, self.stump_weights):
            if stump.feature_idx is not None:
                importance_scores[stump.feature_idx] += abs(weight)
        
        # Normalize to get relative importance
        if np.sum(importance_scores) > 0:
            importance_scores = importance_scores / np.sum(importance_scores)
        
        self.feature_importances_ = importance_scores
    
    def _predict_binary(self, X):
        """
        Make binary predictions by combining all stumps
        Enhanced with better numerical handling
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        num_samples = X.shape[0]
        final_predictions = np.zeros(num_samples)
        
        # Weighted voting: each stump votes, but some votes count more
        for stump, weight in zip(self.stumps, self.stump_weights):
            stump_pred = stump.predict(X)
            final_predictions += weight * stump_pred
        
        # Convert to final binary prediction
        return np.sign(final_predictions)
    
    def predict(self, X):
        """
        Make predictions and convert back to original labels
        Enhanced with better label conversion for heart disease prediction
        """
        binary_preds = self._predict_binary(X)
        
        # Convert binary predictions back to original labels
        original_preds = []
        for pred in binary_preds:
            if pred >= 0:  # Use >= instead of == for better numerical stability
                # Find original label for +1 (disease)
                for orig_label, binary_label in self.label_mapping.items():
                    if binary_label == 1:
                        original_preds.append(orig_label)
                        break
            else:
                # Find original label for -1 (no disease)
                for orig_label, binary_label in self.label_mapping.items():
                    if binary_label == -1:
                        original_preds.append(orig_label)
                        break
        
        return np.array(original_preds)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities using decision function
        Enhanced version with better probability calibration for medical diagnosis
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        num_samples = X.shape[0]
        decision_scores = np.zeros(num_samples)
        
        # Calculate decision function value
        for stump, weight in zip(self.stumps, self.stump_weights):
            stump_pred = stump.predict(X)
            decision_scores += weight * stump_pred
        
        # Convert to probabilities using sigmoid-like function
        # Enhanced with better numerical stability
        decision_scores = np.clip(decision_scores, -250, 250)  # Prevent overflow
        probabilities = 1 / (1 + np.exp(-2 * decision_scores))
        
        # Create probability matrix for both classes
        # probabilities[i] = P(disease | patient i)
        # 1 - probabilities[i] = P(no disease | patient i)
        prob_matrix = np.column_stack([1 - probabilities, probabilities])
        
        return prob_matrix
    
    def feature_importance(self):
        """
        Get feature importance scores for heart disease prediction
        
        Returns:
        --------
        array-like
            Feature importance scores (normalized to sum to 1)
        """
        return self.feature_importances_
    
    def get_training_stats(self):
        """
        Get some stats about the training process
        Useful for debugging and understanding what happened
        """
        return {
            'training_errors': self.training_errors,
            'stump_weights': self.stump_weights,
            'num_stumps': len(self.stumps)
        }
    
    def get_stump_details(self):
        """
        Get details about each decision stump
        Good for understanding what the algorithm learned about heart disease
        """
        details = []
        for i, (stump, weight) in enumerate(zip(self.stumps, self.stump_weights)):
            feature_name = f"feature_{stump.feature_idx}"
            if self.feature_names and stump.feature_idx < len(self.feature_names):
                feature_name = self.feature_names[stump.feature_idx]
            
            detail = {
                'stump_num': i,
                'feature': feature_name,
                'feature_idx': stump.feature_idx,
                'threshold': stump.threshold,
                'polarity': stump.polarity,
                'weight': weight,
                'training_error': self.training_errors[i] if i < len(self.training_errors) else None
            }
            details.append(detail)
        
        return details
    
    def staged_predict(self, X):
        """
        See how predictions change as we add more stumps
        This is useful for visualizing how AdaBoost improves over time
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        num_samples = X.shape[0]
        staged_preds = []
        cumulative_pred = np.zeros(num_samples)
        
        for stump, weight in zip(self.stumps, self.stump_weights):
            stump_pred = stump.predict(X)
            cumulative_pred += weight * stump_pred
            
            # Convert to binary then to original labels
            binary_pred = np.sign(cumulative_pred)
            
            original_pred = []
            for pred in binary_pred:
                if pred >= 0:
                    for orig_label, binary_label in self.label_mapping.items():
                        if binary_label == 1:
                            original_pred.append(orig_label)
                            break
                else:
                    for orig_label, binary_label in self.label_mapping.items():
                        if binary_label == -1:
                            original_pred.append(orig_label)
                            break
            
            staged_preds.append(np.array(original_pred))
        
        return staged_preds
    
    def decision_function(self, X):
        """
        Calculate the decision function values
        Useful for understanding confidence in predictions
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        num_samples = X.shape[0]
        decision_scores = np.zeros(num_samples)
        
        # Calculate weighted sum of stump predictions
        for stump, weight in zip(self.stumps, self.stump_weights):
            stump_pred = stump.predict(X)
            decision_scores += weight * stump_pred
        
        return decision_scores

# Test the implementation with heart disease data
if __name__ == "__main__":
    # Test using heart disease data generation
    
    def create_heart_disease_test_data(n_samples=200, random_state=42):
        """Create realistic heart disease data for testing AdaBoost"""
        np.random.seed(random_state)
        
        # Generate realistic heart disease features
        data = {}
        
        # Age (normally distributed around 54)
        data['age'] = np.clip(np.random.normal(54, 9, n_samples), 29, 77)
        
        # Sex (68% male)
        data['sex'] = np.random.choice([0, 1], n_samples, p=[0.32, 0.68])
        
        # Chest pain type (0-3)
        data['cp'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.47, 0.17, 0.28, 0.08])
        
        # Blood pressure
        data['trestbps'] = np.clip(np.random.normal(131, 17, n_samples), 94, 200)
        
        # Cholesterol
        data['chol'] = np.clip(np.random.normal(246, 51, n_samples), 126, 564)
        
        # Max heart rate (age dependent)
        base_hr = 220 - data['age']
        noise = np.random.normal(0, 23, n_samples)
        data['thalach'] = np.clip(base_hr + noise, 71, 202)
        
        # Exercise angina
        data['exang'] = np.random.choice([0, 1], n_samples, p=[0.67, 0.33])
        
        # ST depression
        data['oldpeak'] = np.clip(np.random.exponential(1.0, n_samples), 0, 6.2)
        
        # Create realistic target based on medical risk factors
        risk_score = (
            (data['age'] - 40) * 0.02 +           # Age factor
            data['sex'] * 0.3 +                   # Male factor
            (data['trestbps'] - 120) * 0.01 +     # BP factor
            (data['chol'] - 200) * 0.001 +        # Cholesterol factor
            (170 - data['thalach']) * 0.01 +      # Heart rate factor (inverted)
            data['exang'] * 0.4 +                 # Exercise angina
            data['oldpeak'] * 0.2 +               # ST depression
            (data['cp'] == 0) * 0.5               # Typical angina
        )
        
        # Add some randomness
        risk_score += np.random.normal(0, 0.3, n_samples)
        
        # Convert to binary target (roughly 55% with disease)
        threshold = np.percentile(risk_score, 45)
        target = (risk_score > threshold).astype(int)
        
        # Create feature matrix
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'exang', 'oldpeak']
        X = np.column_stack([data[name] for name in feature_names])
        
        return X, target, feature_names
    
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
    
    def simple_classification_report(y_true, y_pred):
        """Simple classification report for heart disease"""
        classes = np.unique(np.concatenate([y_true, y_pred]))
        
        print(f"{'Class':>15} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
        print("-" * 55)
        
        for cls in classes:
            # Calculate metrics for this class
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            support = np.sum(y_true == cls)
            
            label = "No Disease" if cls == 0 else "Heart Disease"
            print(f"{label:>15} {precision:>10.3f} {recall:>8.3f} {f1:>8.3f} {support:>8}")
    
    print("Testing AdaBoost Implementation for Heart Disease Prediction")
    print("=" * 65)
    
    # Make heart disease test data
    print("Creating heart disease test dataset...")
    X, y, feature_names = create_heart_disease_test_data(n_samples=200, random_state=42)
    
    print(f"Dataset created: {X.shape[0]} patients, {X.shape[1]} features")
    print(f"Disease prevalence: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = simple_train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Class distribution in training: No Disease={sum(y_train==0)}, Disease={sum(y_train==1)}")
    
    # Train AdaBoost
    print("\nTraining AdaBoost for heart disease prediction...")
    import time
    start_time = time.time()
    
    ada = AdaBoost(num_estimators=50, learning_rate=1.0, random_state=42)
    ada.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training took: {training_time:.2f} seconds")
    
    # Test predictions
    print("\nMaking predictions...")
    y_pred = ada.predict(X_test)
    
    # Evaluate performance
    accuracy = simple_accuracy_score(y_test, y_pred)
    print(f"\nAdaBoost Accuracy: {accuracy:.4f}")
    print("\nDetailed results:")
    simple_classification_report(y_test, y_pred)
    
    # Look at probabilities for first few samples
    probas = ada.predict_proba(X_test[:5])
    print(f"\nPrediction probabilities for first 5 test patients:")
    for i, (prob, true_val, pred_val) in enumerate(zip(probas, y_test[:5], y_pred[:5])):
        print(f"Patient {i+1}: Predicted={pred_val}, Actual={true_val}")
        print(f"  P(No Disease): {prob[0]:.3f}")
        print(f"  P(Heart Disease): {prob[1]:.3f}")
    
    # Feature importance
    importances = ada.feature_importance()
    if importances is not None:
        print(f"\nFeature importance for heart disease prediction:")
        for i, (name, imp) in enumerate(zip(feature_names, importances)):
            print(f"{name:12} {imp:.4f}")
    
    # Some training stats
    stats = ada.get_training_stats()
    print(f"\nTraining statistics:")
    print(f"Number of stumps created: {stats['num_stumps']}")
    print(f"Average training error: {np.mean(stats['training_errors']):.4f}")
    print(f"Average stump weight: {np.mean(stats['stump_weights']):.4f}")
    
    # Details about first few stumps
    stump_info = ada.get_stump_details()
    print(f"\nFirst 5 decision stumps learned:")
    for info in stump_info[:5]:
        print(f"Stump {info['stump_num']}: {info['feature']} "
              f"{'<' if info['polarity'] == 1 else '>='} {info['threshold']:.2f}, "
              f"Weight {info['weight']:.3f}")
    
    print("\n✅ AdaBoost heart disease prediction test completed!")