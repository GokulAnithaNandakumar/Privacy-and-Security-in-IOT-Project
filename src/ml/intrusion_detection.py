import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineering:
    """Feature engineering for IoT sensor data"""

    def __init__(self):
        self.scalers = {}
        self.encoders = {}

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from timestamp"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)

        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        return df

    def create_sensor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sensor-specific features"""
        df = df.copy()

        # Ensure value column is numeric
        df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0)

        # Group by sensor for rolling statistics
        for sensor_type in df['sensor_type'].unique():
            sensor_mask = df['sensor_type'] == sensor_type
            sensor_data = df[sensor_mask].copy()

            if len(sensor_data) > 1:
                # Rolling statistics (window of 5)
                rolling_window = min(5, len(sensor_data))
                df.loc[sensor_mask, f'{sensor_type}_rolling_mean'] = sensor_data['value'].rolling(
                    window=rolling_window, min_periods=1).mean()
                df.loc[sensor_mask, f'{sensor_type}_rolling_std'] = sensor_data['value'].rolling(
                    window=rolling_window, min_periods=1).std().fillna(0)

                # Lag features
                df.loc[sensor_mask, f'{sensor_type}_lag_1'] = sensor_data['value'].shift(1).fillna(sensor_data['value'].iloc[0])

                # Rate of change
                df.loc[sensor_mask, f'{sensor_type}_rate_of_change'] = sensor_data['value'].diff().fillna(0)

        # Fill any remaining NaN values
        df = df.fillna(0)

        return df

    def create_room_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create room-level aggregated features"""
        df = df.copy()

        # Group by room and time window for aggregations
        df['time_window'] = pd.to_datetime(df['timestamp']).dt.floor('1H')

        # Ensure value column is numeric
        df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0)

        try:
            room_features = df.groupby(['room', 'time_window']).agg({
                'value': ['mean', 'std', 'min', 'max', 'count']
            }).reset_index()

            # Flatten column names
            room_features.columns = ['room', 'time_window'] + [f'room_{col[1]}' for col in room_features.columns[2:]]

            # Merge back to original dataframe
            df = df.merge(room_features, on=['room', 'time_window'], how='left')

            # Fill NaN values for numeric columns only
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                df[col] = df[col].fillna(0)

        except Exception as e:
            print(f"Warning: Could not create room features: {e}")
            # Continue without room features if there's an issue

        return df

    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        df = df.copy()
        categorical_columns = ['room', 'sensor_type', 'sensor_id']

        for col in categorical_columns:
            if col in df.columns:
                if fit:
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                        df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
                    else:
                        df[f'{col}_encoded'] = self.encoders[col].transform(df[col].astype(str))
                else:
                    if col in self.encoders:
                        # Handle unseen labels
                        unique_labels = set(self.encoders[col].classes_)
                        df[col] = df[col].apply(lambda x: x if x in unique_labels else 'unknown')
                        df[f'{col}_encoded'] = self.encoders[col].transform(df[col].astype(str))

        return df

    def scale_features(self, df: pd.DataFrame, feature_columns: List[str], fit: bool = True) -> pd.DataFrame:
        """Scale numerical features"""
        df = df.copy()

        for col in feature_columns:
            if col in df.columns:
                if fit:
                    if col not in self.scalers:
                        self.scalers[col] = StandardScaler()
                        df[f'{col}_scaled'] = self.scalers[col].fit_transform(df[[col]])
                    else:
                        df[f'{col}_scaled'] = self.scalers[col].transform(df[[col]])
                else:
                    if col in self.scalers:
                        df[f'{col}_scaled'] = self.scalers[col].transform(df[[col]])

        return df

    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        print("Creating time features...")
        df = self.create_time_features(df)

        print("Creating sensor features...")
        df = self.create_sensor_features(df)

        print("Creating room features...")
        df = self.create_room_features(df)

        print("Encoding categorical features...")
        df = self.encode_categorical_features(df, fit=fit)

        # Define numerical features to scale
        numerical_features = ['value', 'hour', 'day_of_week', 'month']

        # Add rolling features for each sensor type
        for sensor_type in df['sensor_type'].unique():
            numerical_features.extend([
                f'{sensor_type}_rolling_mean',
                f'{sensor_type}_rolling_std',
                f'{sensor_type}_lag_1',
                f'{sensor_type}_rate_of_change'
            ])

        # Add room aggregation features
        room_features = [col for col in df.columns if col.startswith('room_')]
        numerical_features.extend(room_features)

        # Remove duplicates and filter existing columns
        numerical_features = list(set([col for col in numerical_features if col in df.columns]))

        print("Scaling numerical features...")
        df = self.scale_features(df, numerical_features, fit=fit)

        return df


class AnomalyDetector:
    """Anomaly detection for IoT sensor data"""

    def __init__(self, method: str = 'isolation_forest'):
        """
        Initialize anomaly detector

        Args:
            method: 'isolation_forest', 'one_class_svm', or 'dbscan'
        """
        self.method = method
        self.model = None
        self.feature_columns = None
        self.feature_engineer = FeatureEngineering()

        # Initialize model based on method
        if method == 'isolation_forest':
            self.model = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
        elif method == 'one_class_svm':
            self.model = OneClassSVM(gamma='auto', nu=0.1)
        elif method == 'dbscan':
            self.model = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError("Method must be 'isolation_forest', 'one_class_svm', or 'dbscan'")

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare data for training"""
        # Apply feature engineering
        df_features = self.feature_engineer.prepare_features(df, fit=True)

        # Select feature columns (numerical features only)
        feature_columns = [col for col in df_features.columns
                          if col.endswith('_scaled') or col.endswith('_encoded') or
                          col in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                                 'is_weekend', 'is_night', 'is_business_hours']]

        self.feature_columns = feature_columns

        # Get feature matrix
        X = df_features[feature_columns].values

        # Handle any remaining NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, feature_columns

    def train(self, df: pd.DataFrame) -> None:
        """Train the anomaly detection model"""
        print(f"Training {self.method} model...")

        X, feature_columns = self.prepare_training_data(df)

        if self.method in ['isolation_forest', 'one_class_svm']:
            self.model.fit(X)
            print(f"Model trained on {X.shape[0]} samples with {X.shape[1]} features")
        elif self.method == 'dbscan':
            labels = self.model.fit_predict(X)
            print(f"DBSCAN clustering completed. Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict anomalies in new data"""
        if self.model is None or self.feature_columns is None:
            raise ValueError("Model must be trained before prediction")

        # Apply feature engineering (without fitting)
        df_features = self.feature_engineer.prepare_features(df, fit=False)

        # Get feature matrix
        X = df_features[self.feature_columns].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if self.method in ['isolation_forest', 'one_class_svm']:
            predictions = self.model.predict(X)
            # Convert to binary (1 for normal, 0 for anomaly)
            return (predictions == 1).astype(int)
        elif self.method == 'dbscan':
            labels = self.model.fit_predict(X)
            # -1 indicates anomaly in DBSCAN
            return (labels != -1).astype(int)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Get anomaly scores"""
        if self.method != 'isolation_forest':
            raise ValueError("Probability scores only available for Isolation Forest")

        if self.model is None or self.feature_columns is None:
            raise ValueError("Model must be trained before prediction")

        # Apply feature engineering
        df_features = self.feature_engineer.prepare_features(df, fit=False)

        # Get feature matrix
        X = df_features[self.feature_columns].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Get anomaly scores
        scores = self.model.decision_function(X)

        # Convert to probabilities (higher score = more normal)
        # Normalize to [0, 1] range
        min_score, max_score = scores.min(), scores.max()
        if max_score > min_score:
            normalized_scores = (scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.ones_like(scores)

        return normalized_scores

    def save_model(self, filepath: str) -> None:
        """Save trained model"""
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'method': self.method,
            'feature_engineer': self.feature_engineer
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.method = model_data['method']
        self.feature_engineer = model_data['feature_engineer']
        print(f"Model loaded from {filepath}")


class IntrusionDetectionSystem:
    """Complete intrusion detection system for IoT data"""

    def __init__(self, models: List[str] = None):
        """
        Initialize intrusion detection system

        Args:
            models: List of models to use for ensemble detection
        """
        if models is None:
            models = ['isolation_forest', 'one_class_svm']

        self.models = {}
        self.model_names = models

        for model_name in models:
            self.models[model_name] = AnomalyDetector(method=model_name)

    def train_models(self, training_data: pd.DataFrame) -> None:
        """Train all models in the ensemble"""
        print("Training intrusion detection models...")

        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            model.train(training_data)

    def detect_intrusions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect intrusions using ensemble of models"""
        results = data.copy()

        # Get predictions from each model
        model_predictions = {}
        model_scores = {}

        for model_name, model in self.models.items():
            predictions = model.predict(data)
            model_predictions[f'{model_name}_prediction'] = predictions

            # Get scores if available
            if model_name == 'isolation_forest':
                scores = model.predict_proba(data)
                model_scores[f'{model_name}_score'] = scores

        # Add predictions to results
        for col, pred in model_predictions.items():
            results[col] = pred

        for col, score in model_scores.items():
            results[col] = score

        # Ensemble prediction (majority vote)
        prediction_columns = [col for col in results.columns if col.endswith('_prediction')]
        if prediction_columns:
            results['ensemble_prediction'] = results[prediction_columns].mean(axis=1)
            results['is_intrusion'] = (results['ensemble_prediction'] < 0.5).astype(int)

        # Calculate confidence score
        if model_scores:
            score_columns = [col for col in results.columns if col.endswith('_score')]
            results['confidence_score'] = results[score_columns].mean(axis=1)

        return results

    def analyze_intrusions(self, detection_results: pd.DataFrame) -> Dict[str, Any]:
        """Analyze detected intrusions"""
        analysis = {}

        if 'is_intrusion' in detection_results.columns:
            intrusions = detection_results[detection_results['is_intrusion'] == 1]

            analysis['total_intrusions'] = len(intrusions)
            analysis['intrusion_rate'] = len(intrusions) / len(detection_results)

            if len(intrusions) > 0:
                # Analyze by room
                analysis['intrusions_by_room'] = intrusions['room'].value_counts().to_dict()

                # Analyze by sensor type
                analysis['intrusions_by_sensor_type'] = intrusions['sensor_type'].value_counts().to_dict()

                # Analyze by time
                intrusions['hour'] = pd.to_datetime(intrusions['timestamp']).dt.hour
                analysis['intrusions_by_hour'] = intrusions['hour'].value_counts().to_dict()

                # Most suspicious sensors
                analysis['most_suspicious_sensors'] = intrusions['sensor_id'].value_counts().head(5).to_dict()

        return analysis

    def save_system(self, directory: str) -> None:
        """Save the entire detection system"""
        import os
        os.makedirs(directory, exist_ok=True)

        for model_name, model in self.models.items():
            filepath = os.path.join(directory, f"{model_name}_model.pkl")
            model.save_model(filepath)

    def load_system(self, directory: str) -> None:
        """Load the entire detection system"""
        import os

        for model_name in self.model_names:
            filepath = os.path.join(directory, f"{model_name}_model.pkl")
            if os.path.exists(filepath):
                if model_name not in self.models:
                    self.models[model_name] = AnomalyDetector(method=model_name)
                self.models[model_name].load_model(filepath)


class IntrusionVisualizer:
    """Visualization for intrusion detection results"""

    @staticmethod
    def plot_intrusion_timeline(detection_results: pd.DataFrame, figsize: Tuple[int, int] = (15, 8)):
        """Plot intrusion detection timeline"""
        plt.figure(figsize=figsize)

        # Convert timestamp to datetime
        detection_results['datetime'] = pd.to_datetime(detection_results['timestamp'])

        # Plot normal vs intrusion points
        normal_data = detection_results[detection_results['is_intrusion'] == 0]
        intrusion_data = detection_results[detection_results['is_intrusion'] == 1]

        plt.scatter(normal_data['datetime'], normal_data['value'],
                   alpha=0.6, c='blue', label='Normal', s=10)
        plt.scatter(intrusion_data['datetime'], intrusion_data['value'],
                   alpha=0.8, c='red', label='Intrusion', s=30, marker='x')

        plt.xlabel('Time')
        plt.ylabel('Sensor Value')
        plt.title('Intrusion Detection Timeline')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_intrusion_heatmap(detection_results: pd.DataFrame, figsize: Tuple[int, int] = (12, 8)):
        """Plot intrusion heatmap by room and time"""
        detection_results['datetime'] = pd.to_datetime(detection_results['timestamp'])
        detection_results['hour'] = detection_results['datetime'].dt.hour
        detection_results['date'] = detection_results['datetime'].dt.date

        # Create pivot table
        intrusion_pivot = detection_results.groupby(['room', 'hour'])['is_intrusion'].sum().unstack(fill_value=0)

        plt.figure(figsize=figsize)
        sns.heatmap(intrusion_pivot, annot=True, cmap='Reds', fmt='d')
        plt.title('Intrusion Heatmap by Room and Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Room')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_model_comparison(detection_results: pd.DataFrame, figsize: Tuple[int, int] = (10, 6)):
        """Compare different model predictions"""
        prediction_columns = [col for col in detection_results.columns if col.endswith('_prediction')]

        if len(prediction_columns) > 1:
            plt.figure(figsize=figsize)

            # Calculate intrusion rates for each model
            intrusion_rates = {}
            for col in prediction_columns:
                model_name = col.replace('_prediction', '')
                intrusion_rate = (detection_results[col] == 0).mean()  # 0 means intrusion
                intrusion_rates[model_name] = intrusion_rate

            # Plot comparison
            plt.bar(intrusion_rates.keys(), intrusion_rates.values())
            plt.xlabel('Model')
            plt.ylabel('Intrusion Detection Rate')
            plt.title('Model Comparison: Intrusion Detection Rates')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    # Example usage
    from ..sensors.sensor_simulator import IoTSensorSimulator

    # Generate sample data
    print("Generating sensor data...")
    simulator = IoTSensorSimulator(num_rooms=3)
    sensor_data = simulator.generate_full_dataset(duration_hours=48)

    # Split data for training and testing
    train_data = sensor_data[:int(0.7 * len(sensor_data))]
    test_data = sensor_data[int(0.7 * len(sensor_data)):]

    # Initialize and train intrusion detection system
    print("\nInitializing intrusion detection system...")
    ids = IntrusionDetectionSystem(['isolation_forest', 'one_class_svm'])
    ids.train_models(train_data)

    # Detect intrusions in test data
    print("\nDetecting intrusions...")
    detection_results = ids.detect_intrusions(test_data)

    # Analyze results
    analysis = ids.analyze_intrusions(detection_results)
    print(f"\nIntrusion Analysis:")
    print(f"Total intrusions detected: {analysis.get('total_intrusions', 0)}")
    print(f"Intrusion rate: {analysis.get('intrusion_rate', 0):.3f}")

    if analysis.get('intrusions_by_room'):
        print(f"Intrusions by room: {analysis['intrusions_by_room']}")

    # Save detection results
    detection_results.to_csv("../data/intrusion_detection_results.csv", index=False)
    print("\nDetection results saved to ../data/intrusion_detection_results.csv")
