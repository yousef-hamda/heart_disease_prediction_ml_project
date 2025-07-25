import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

class HeartDiseaseDataCollector:
    """
    A comprehensive class for collecting and preparing heart disease data
    This class handles data collection from UCI repository and creates features for ML models
    """
    
    def __init__(self):
        # Base URL for UCI Heart Disease dataset
        self.base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"
        
        # Column names for the dataset
        self.column_names = [
            'age',          # Age in years
            'sex',          # Sex (1 = male; 0 = female)
            'cp',           # Chest pain type (1-4)
            'trestbps',     # Resting blood pressure (mm Hg)
            'chol',         # Serum cholestoral (mg/dl)
            'fbs',          # Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
            'restecg',      # Resting electrocardiographic results (0-2)
            'thalach',      # Maximum heart rate achieved
            'exang',        # Exercise induced angina (1 = yes; 0 = no)
            'oldpeak',      # ST depression induced by exercise relative to rest
            'slope',        # Slope of the peak exercise ST segment (1-3)
            'ca',           # Number of major vessels (0-3) colored by flourosopy
            'thal',         # Thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
            'target'        # Diagnosis of heart disease (0 = no disease, 1-4 = disease)
        ]
        
    def fetch_heart_disease_data(self, location='cleveland'):
        """
        Fetch heart disease data from UCI repository
        
        Parameters:
        -----------
        location : str
            Location of the dataset ('cleveland', 'hungarian', 'switzerland', 'va')
            Cleveland has the most complete data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing heart disease data
        """
        print(f"Fetching heart disease data from {location}...")
        
        try:
            # Construct URL for the specific location
            if location == 'cleveland':
                url = f"{self.base_url}/processed.cleveland.data"
            elif location == 'hungarian':
                url = f"{self.base_url}/processed.hungarian.data"
            elif location == 'switzerland':
                url = f"{self.base_url}/processed.switzerland.data"
            elif location == 'va':
                url = f"{self.base_url}/processed.va.data"
            else:
                raise ValueError(f"Unknown location: {location}")
            
            # Make HTTP request
            response = requests.get(url)
            response.raise_for_status()
            
            # Save data to temporary file and read with pandas
            temp_filename = f'temp_{location}_data.csv'
            with open(temp_filename, 'w') as f:
                f.write(response.text)
            
            # Read data with pandas
            df = pd.read_csv(temp_filename, names=self.column_names, na_values='?')
            
            # Clean up temporary file
            import os
            os.remove(temp_filename)
            
            # Add location identifier
            df['location'] = location
            
            print(f"Successfully fetched {len(df)} records from {location}")
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from {location}: {e}")
            return None
        except Exception as e:
            print(f"Error processing data from {location}: {e}")
            return None
    
    def create_fallback_data(self):
        """
        Create realistic simulated heart disease data if online data fails
        Based on actual heart disease statistics and patterns
        """
        print("Creating simulated heart disease data...")
        
        np.random.seed(42)
        n_samples = 300  # Similar to original Cleveland dataset size
        
        # Generate realistic patient data
        data = {}
        
        # Age: normally distributed around 54 years (realistic for heart disease studies)
        data['age'] = np.clip(np.random.normal(54, 9, n_samples), 29, 77).astype(int)
        
        # Sex: roughly 68% male (typical for heart disease studies)
        data['sex'] = np.random.choice([0, 1], n_samples, p=[0.32, 0.68])
        
        # Chest pain type: 4 types, with type 0 (typical angina) being most common in disease
        data['cp'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.47, 0.17, 0.28, 0.08])
        
        # Resting blood pressure: normally distributed around 131 mmHg
        data['trestbps'] = np.clip(np.random.normal(131, 17, n_samples), 94, 200).astype(int)
        
        # Cholesterol: normally distributed around 246 mg/dl
        data['chol'] = np.clip(np.random.normal(246, 51, n_samples), 126, 564).astype(int)
        
        # Fasting blood sugar: about 15% have FBS > 120
        data['fbs'] = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
        
        # Resting ECG: mostly normal (0), some abnormalities
        data['restecg'] = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.48, 0.02])
        
        # Maximum heart rate: inversely related to age
        base_hr = 220 - data['age']  # Standard formula
        noise = np.random.normal(0, 23, n_samples)
        data['thalach'] = np.clip(base_hr + noise, 71, 202).astype(int)
        
        # Exercise induced angina: about 33% have it
        data['exang'] = np.random.choice([0, 1], n_samples, p=[0.67, 0.33])
        
        # ST depression: mostly around 0-2
        data['oldpeak'] = np.clip(np.random.exponential(1.0, n_samples), 0, 6.2)
        
        # ST slope: 3 types
        data['slope'] = np.random.choice([0, 1, 2], n_samples, p=[0.21, 0.14, 0.65])
        
        # Number of major vessels: 0-3
        data['ca'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.54, 0.19, 0.17, 0.10])
        
        # Thalassemia: 3 main types
        data['thal'] = np.random.choice([3, 6, 7], n_samples, p=[0.55, 0.18, 0.27])
        
        # Create target based on realistic risk factors
        # Higher risk with: older age, male, high BP, high cholesterol, low max HR
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
        
        # Convert to binary target (roughly 55% with disease, like original data)
        threshold = np.percentile(risk_score, 45)
        data['target'] = (risk_score > threshold).astype(int)
        
        # Add location identifier
        data['location'] = 'simulated'
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        print(f"✅ Created {len(df)} simulated samples")
        print(f"Target distribution: Disease={sum(df['target'])}, Healthy={len(df)-sum(df['target'])}")
        
        return df
    
    def clean_and_prepare_data(self, df):
        """
        Clean and prepare the heart disease data for machine learning
        Remove missing values and ensure proper data types
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Raw heart disease dataset
            
        Returns:
        --------
        tuple
            (X, y, clean_df) - Features, target, and clean dataset
        """
        print("\n" + "="*60)
        print("CLEANING AND PREPARING DATA FOR MACHINE LEARNING")
        print("="*60)
        
        # Remove rows with missing values
        initial_rows = len(df)
        df_clean = df.dropna()
        final_rows = len(df_clean)
        print(f"Removed {initial_rows - final_rows} rows with missing values")
        
        # Convert target to binary (0 = no disease, 1 = disease)
        # Original dataset has 0-4, where 0 = no disease, 1-4 = disease
        df_clean['target'] = (df_clean['target'] > 0).astype(int)
        
        # Select features for machine learning (exclude location and target)
        feature_columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        
        # Ensure all features are present
        available_features = [col for col in feature_columns if col in df_clean.columns]
        print(f"Selected {len(available_features)} features for modeling")
        
        # Create feature matrix and target vector
        X = df_clean[available_features].copy()
        y = df_clean['target'].copy()
        
        # Ensure proper data types
        X = X.astype(float)
        y = y.astype(int)
        
        print(f"Final dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Target distribution: Disease={sum(y)} ({sum(y)/len(y)*100:.1f}%), "
              f"Healthy={len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
        
        # Display feature information
        print(f"\nFeature summary:")
        for i, feature in enumerate(available_features):
            min_val = X[feature].min()
            max_val = X[feature].max()
            mean_val = X[feature].mean()
            print(f"  {feature:12} - Range: [{min_val:6.1f}, {max_val:6.1f}], Mean: {mean_val:6.1f}")
        
        return X, y, df_clean
    
    def collect_all_data(self, use_online_data=True):
        """
        Main method to collect and process heart disease data
        This orchestrates the entire data collection and preparation pipeline
        
        Parameters:
        -----------
        use_online_data : bool
            Whether to try fetching online data or use simulated data
            
        Returns:
        --------
        pandas.DataFrame
            Complete dataset ready for machine learning, or None if collection fails
        """
        print("="*60)
        print("STARTING HEART DISEASE DATA COLLECTION")
        print("="*60)
        
        if use_online_data:
            print("Attempting to fetch real data from UCI repository...")
            
            # Try Cleveland data first (most complete)
            df = self.fetch_heart_disease_data('cleveland')
            
            if df is None or len(df) < 50:
                print("Cleveland data failed, trying other locations...")
                
                # Try other locations
                for location in ['hungarian', 'switzerland', 'va']:
                    df_temp = self.fetch_heart_disease_data(location)
                    if df_temp is not None and len(df_temp) > 50:
                        if df is None:
                            df = df_temp
                        else:
                            df = pd.concat([df, df_temp], ignore_index=True)
            
            if df is None or len(df) < 50:
                print("❌ Online data collection failed, using simulated data")
                df = self.create_fallback_data()
            else:
                print(f"✅ Successfully collected {len(df)} real samples")
        else:
            print("Using simulated heart disease data...")
            df = self.create_fallback_data()
        
        return df
    
    def get_feature_descriptions(self):
        """
        Get detailed descriptions of all features in the dataset
        Useful for interpretation and reporting
        
        Returns:
        --------
        dict
            Dictionary mapping feature names to descriptions
        """
        descriptions = {
            'age': 'Age in years (29-77)',
            'sex': 'Sex (1 = male, 0 = female)',
            'cp': 'Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)',
            'trestbps': 'Resting blood pressure in mm Hg on admission to hospital',
            'chol': 'Serum cholesterol in mg/dl',
            'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)',
            'restecg': 'Resting electrocardiographic results (0 = normal, 1 = ST-T abnormality, 2 = left ventricular hypertrophy)',
            'thalach': 'Maximum heart rate achieved during exercise test',
            'exang': 'Exercise induced angina (1 = yes, 0 = no)',
            'oldpeak': 'ST depression induced by exercise relative to rest',
            'slope': 'Slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping)',
            'ca': 'Number of major vessels (0-3) colored by fluoroscopy',
            'thal': 'Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)',
            'target': 'Heart disease diagnosis (0 = no disease, 1 = disease present)'
        }
        return descriptions

# Example usage and testing
if __name__ == "__main__":
    # Create data collector instance
    collector = HeartDiseaseDataCollector()
    
    # Display feature descriptions
    print("Heart Disease Dataset Features:")
    print("=" * 50)
    descriptions = collector.get_feature_descriptions()
    for feature, desc in descriptions.items():
        if feature != 'target':  # Don't show target in features
            print(f"{feature:12} - {desc}")
    
    print(f"\nTarget variable:")
    print(f"{'target':12} - {descriptions['target']}")
    
    # Collect heart disease data
    print(f"\n\nStarting heart disease data collection...")
    data = collector.collect_all_data(use_online_data=True)
    
    if data is not None:
        print(f"\n✅ Successfully collected {len(data)} records")
        
        # Clean and prepare data for machine learning
        X, y, clean_data = collector.clean_and_prepare_data(data)
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"Features shape: {X.shape}")
        print(f"Target samples: {len(y)}")
        print(f"Ready for machine learning!")
        
        # Show some statistics
        print(f"\nDataset statistics:")
        print(f"Average age: {X['age'].mean():.1f} years")
        print(f"Male patients: {X['sex'].sum()} ({X['sex'].mean()*100:.1f}%)")
        print(f"Disease prevalence: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
        
    else:
        print("\n❌ Data collection failed")