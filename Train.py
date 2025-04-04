import os
import re
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from supabase import create_client, Client
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from multiprocessing import Pool, cpu_count

class PhishingDetector:
    def __init__(self):
        # Initialize Supabase client
        self.supabase = create_client(
            "https://ydhicwwzijljkrmihjcp.supabase.co",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlkaGljd3d6aWpsamtybWloamNwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDM2NzMwMTEsImV4cCI6MjA1OTI0OTAxMX0.ij_yuhiu_USjh_2wPpLDSewnU4c4alTfjsnGYVq2Wb4"
        )

        # Initialize preprocessing components
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler = StandardScaler()
        
        # Using Random Forest with optimized parameters for large datasets
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            class_weight={0: 1, 1: 4},  # Increased weight for bad URLs
            random_state=42,
            n_jobs=-1  # Use all available cores
        )

        # Feature configuration
        self.numeric_features = ['url_length', 'dots_count', 'digits_count', 
                               'special_chars_count', 'path_depth', 'avg_token_length']
        self.categorical_features = ['has_http', 'has_www']

    @staticmethod
    def _extract_url_features(url):
        """Feature extraction helper - static method for parallel processing"""
        url = str(url).lower()
        
        # Calculate average token length
        tokens = re.split(r'[/.-_]', url)
        valid_tokens = [t for t in tokens if t]
        avg_token_length = np.mean([len(t) for t in valid_tokens]) if valid_tokens else 0
        
        return {
            'url_length': len(url),
            'dots_count': url.count('.'),
            'digits_count': sum(map(str.isdigit, url)),
            'special_chars_count': len(re.findall(r'[^a-z0-9.-_/]', url)),
            'path_depth': url.count('/'),
            'has_http': int(url.startswith('http://')),
            'has_www': int('www.' in url),
            'avg_token_length': avg_token_length,
            'tld': int(url.split('.')[-1] in {'com', 'net', 'org'}),
            'shortened': int('bit.ly' in url or 'goo.gl' in url),
            'hex_chars': int(bool(re.search(r'%[0-9a-f]{2}', url)))
        }

    def extract_features_batch(self, urls, batch_size=10000):
        """Process URLs in batches to prevent memory issues"""
        all_features = []
        
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i+batch_size]
            
            # Process each batch
            with Pool(processes=min(cpu_count(), 8)) as pool:  # Limit to 8 cores max
                batch_features = list(pool.map(self._extract_url_features, batch))
            
            all_features.extend(batch_features)
            print(f"Processed {min(i+batch_size, len(urls))}/{len(urls)} URLs")
            
        return pd.DataFrame(all_features)

    def load_data(self):
        """Load data with column verification"""
        try:
            # Fetch sample data to verify columns
            test_response = self.supabase.table("urls")\
                .select("url, type")\
                .limit(1)\
                .execute()
            
            if not test_response.data:
                print("No data found in table")
                return pd.DataFrame()
            
            # Verify column names
            first_record = test_response.data[0]
            if 'url' not in first_record or 'type' not in first_record:
                print("Missing required columns. Actual columns:", first_record.keys())
                return pd.DataFrame()

            # Proceed with full data load
            all_records = []
            page_size = 50000
            offset = 0
            
            print("Loading data from Supabase...")
            while True:
                response = self.supabase.table("urls")\
                    .select("url, type")\
                    .range(offset, offset + page_size - 1)\
                    .execute()
                
                if not response.data:
                    break
                
                all_records.extend(response.data)
                offset += len(response.data)
                print(f"Loaded {offset} records...")

            full_df = pd.DataFrame(all_records)
            
            # Clean column names
            full_df.rename(columns={
                'url': 'URL',
                'type': 'Type'
            }, inplace=True)
            
            # Validate data structure
            if 'Type' not in full_df.columns:
                print("Error: 'Type' column not found after rename")
                return pd.DataFrame()
            
            # Map labels
            full_df['Label'] = full_df['Type'].apply(
                lambda x: 0 if str(x).lower() in ['good', 'benign'] else 1
            )
            
            # Sample 10% of the data
            sampled_df = full_df.sample(frac=0.1, random_state=42)
            
            print("Data loaded successfully")
            return sampled_df[['URL', 'Label']]

        except Exception as e:
            print(f"Data loading failed: {str(e)}")
            return pd.DataFrame()

    def preprocess_data(self, df, test_size=0.2):
        """Full preprocessing pipeline with efficient feature extraction"""
        X = df['URL'].values
        y = df['Label'].values
        
        print("Splitting data into train and test sets...")
        # Split before feature extraction
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=42
        )
        
        print(f"Extracting features from {len(X_train)} training URLs...")
        train_features = self.extract_features_batch(X_train)
        
        print(f"Extracting features from {len(X_test)} test URLs...")
        test_features = self.extract_features_batch(X_test)
        
        print("\nFeature summary:")
        print(train_features.describe())
        
        return train_features, test_features, y_train, y_test

    def prepare_features(self, train_df, test_df):
        """Feature engineering with proper train/test separation"""
        print("Preparing features...")
        # Fit transformers on training data only
        self.encoder.fit(train_df[self.categorical_features])
        self.scaler.fit(train_df[self.numeric_features])
        
        # Transform both sets
        train_cat = self.encoder.transform(train_df[self.categorical_features])
        train_num = self.scaler.transform(train_df[self.numeric_features])
        
        test_cat = self.encoder.transform(test_df[self.categorical_features])
        test_num = self.scaler.transform(test_df[self.numeric_features])
        
        # Combine features
        X_train = np.hstack([train_num, train_cat])
        X_test = np.hstack([test_num, test_cat])
        
        return X_train, X_test

    def train_model(self, X_train, y_train):
        """Training with SMOTEENN for class balance"""
        print(f"Number of unique classes in training data: {len(np.unique(y_train))}")
        print(f"Class distribution: {np.bincount(y_train)}")
        
        print("Applying SMOTEENN for class balance...")
        smote_enn = SMOTEENN(random_state=42)
        X_res, y_res = smote_enn.fit_resample(X_train, y_train)
        
        print(f"Data after balancing: {X_res.shape[0]} samples")
        print(f"Balanced class distribution: {np.bincount(y_res)}")
        
        print("Training model...")
        self.model.fit(X_res, y_res)

    def evaluate_model(self, X_test, y_test):
        """Basic model evaluation"""
        print("Evaluating model...")
        y_pred = self.model.predict(X_test)
        
        print("\nEvaluation Metrics:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    def save_model(self, model_name="phishing_detector", directory="models"):
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, f"{model_name}.joblib")
        print(f"Saving model to {model_path}...")
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'encoder': self.encoder,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features
        }, model_path)
        return model_path

    def load_model(self, model_path):
        print(f"Loading model from {model_path}...")
        data = joblib.load(model_path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.encoder = data['encoder']
        self.numeric_features = data.get('numeric_features', self.numeric_features)
        self.categorical_features = data.get('categorical_features', self.categorical_features)

    def predict(self, urls, good_threshold=0.85, bad_threshold=0.90):
        """Prediction with thresholds for good, bad, and uncertain classifications"""
        if isinstance(urls, str):
            urls = [urls]
            
        # Extract features
        features = pd.DataFrame([self._extract_url_features(url) for url in urls])
        
        # Transform features
        cat_features = self.encoder.transform(features[self.categorical_features])
        num_features = self.scaler.transform(features[self.numeric_features])
        
        # Combine features
        X = np.hstack([num_features, cat_features])
        
        # Predict probabilities
        probas = self.model.predict_proba(X)
        return [
            {
                'url': url,
                'prediction': self._get_prediction(probas[i], good_threshold, bad_threshold),
                'confidence': max(probas[i])
            }
            for i, url in enumerate(urls)
        ]

    def _get_prediction(self, proba, good_thresh, bad_thresh):
        """Determine prediction based on thresholds"""
        if proba[0] > good_thresh:
            return 'good'
        elif proba[1] > bad_thresh:
            return 'bad'
        return 'uncertain'

    def explain_prediction(self, url):
        features = self._extract_url_features(url)
        explanation = {
            'raw_features': features,
            'feature_importances': dict(zip(
                self.numeric_features + list(self.encoder.get_feature_names_out()),
                self.model.feature_importances_
            ))
        }
        return explanation

    def check_existing_url(self, url):
        try:
            response = self.supabase.table("urls") \
                .select("type") \
                .eq("url", url) \
                .execute()
            return response.data[0]['type'] if response.data else None
        except Exception as e:
            print(f"Database check error: {str(e)}")
            return None

if __name__ == "__main__":
    detector = PhishingDetector()
    
    # Load ALL data
    raw_data = detector.load_data()
    
    if not raw_data.empty:
        print("Data sample:")
        print(raw_data.head())
        print("\nLabel distribution:")
        print(raw_data['Label'].value_counts())
        print(f"Total records: {len(raw_data)}")
        
    # Preprocess with proper train/test split
    train_feats, test_feats, y_train, y_test = detector.preprocess_data(raw_data)
    
    # Prepare features
    X_train, X_test = detector.prepare_features(train_feats, test_feats)
    
    # Train and evaluate
    detector.train_model(X_train, y_train)
    detector.evaluate_model(X_test, y_test)
    
    # Save model
    model_path = detector.save_model()
    print(f"\nModel saved to: {model_path}")
    
    # Example prediction
    test_urls = [
        "https://www.google.com",
        "http://phishing-example.com/login.php?user=123",
        "https://legit-but-risky.site"
    ]
    print("\nTesting model with example URLs:")
    predictions = detector.predict(test_urls)
    for result in predictions:
        print(f"URL: {result['url']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("---")