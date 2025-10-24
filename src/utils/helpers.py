import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('../data/iot_simulation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('IoTSimulation')

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        # Return default configuration
        return {
            "simulation": {
                "num_rooms": 5,
                "duration_hours": 168,
                "sensor_types": ["motion", "light", "temperature", "humidity", "air_quality"]
            },
            "privacy": {
                "epsilon": 1.0,
                "k_anonymity": 5,
                "enable_encryption": True
            },
            "blockchain": {
                "difficulty": 4,
                "max_transactions_per_block": 10
            },
            "ml": {
                "models": ["isolation_forest", "one_class_svm"],
                "train_test_split": 0.7
            }
        }

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to JSON file"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def create_project_directories(base_path: str) -> None:
    """Create necessary project directories"""
    directories = [
        'data',
        'models',
        'logs',
        'outputs',
        'visualization/static',
        'visualization/templates'
    ]

    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        os.makedirs(dir_path, exist_ok=True)

def validate_sensor_data(data: pd.DataFrame) -> Dict[str, Any]:
    """Validate sensor data format and content"""
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }

    # Check required columns
    required_columns = ['timestamp', 'room', 'sensor_type', 'value', 'sensor_id']
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Missing required columns: {missing_columns}")

    # Check data types
    if 'timestamp' in data.columns:
        try:
            pd.to_datetime(data['timestamp'])
        except:
            validation_results['errors'].append("Invalid timestamp format")

    if 'value' in data.columns:
        if not pd.api.types.is_numeric_dtype(data['value']):
            validation_results['warnings'].append("Value column is not numeric")

    # Check for missing values
    null_counts = data.isnull().sum()
    if null_counts.any():
        validation_results['warnings'].append(f"Found null values: {null_counts[null_counts > 0].to_dict()}")

    # Check data consistency
    if len(data) == 0:
        validation_results['is_valid'] = False
        validation_results['errors'].append("Dataset is empty")

    return validation_results

def generate_data_summary(data: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive data summary"""
    summary = {}

    # Basic statistics
    summary['basic_stats'] = {
        'total_records': len(data),
        'time_range': {
            'start': data['timestamp'].min() if 'timestamp' in data.columns else None,
            'end': data['timestamp'].max() if 'timestamp' in data.columns else None
        },
        'unique_rooms': data['room'].nunique() if 'room' in data.columns else 0,
        'unique_sensors': data['sensor_id'].nunique() if 'sensor_id' in data.columns else 0,
        'sensor_types': data['sensor_type'].unique().tolist() if 'sensor_type' in data.columns else []
    }

    # Value statistics by sensor type
    if 'sensor_type' in data.columns and 'value' in data.columns:
        summary['sensor_stats'] = {}
        for sensor_type in data['sensor_type'].unique():
            sensor_data = data[data['sensor_type'] == sensor_type]['value']
            summary['sensor_stats'][sensor_type] = {
                'count': len(sensor_data),
                'mean': sensor_data.mean(),
                'std': sensor_data.std(),
                'min': sensor_data.min(),
                'max': sensor_data.max(),
                'median': sensor_data.median()
            }

    # Room statistics
    if 'room' in data.columns:
        summary['room_stats'] = data['room'].value_counts().to_dict()

    return summary

def export_results(results: Dict[str, Any], output_path: str, format: str = 'json') -> None:
    """Export results to file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if format.lower() == 'json':
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        # Recursively convert numpy types
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(item) for item in obj]
            else:
                return convert_numpy(obj)

        converted_results = recursive_convert(results)

        with open(output_path, 'w') as f:
            json.dump(converted_results, f, indent=2, default=str)

    elif format.lower() == 'csv':
        if isinstance(results, pd.DataFrame):
            results.to_csv(output_path, index=False)
        else:
            # Convert dict to DataFrame if possible
            try:
                df = pd.DataFrame(results)
                df.to_csv(output_path, index=False)
            except:
                raise ValueError("Cannot convert results to CSV format")

def calculate_performance_metrics(detection_results: pd.DataFrame,
                                ground_truth_column: str = None) -> Dict[str, float]:
    """Calculate performance metrics for intrusion detection"""
    metrics = {}

    if ground_truth_column and ground_truth_column in detection_results.columns:
        # If we have ground truth, calculate standard metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        y_true = detection_results[ground_truth_column]
        y_pred = detection_results['is_intrusion']

        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    else:
        # Calculate basic detection statistics
        total_readings = len(detection_results)
        intrusions_detected = detection_results['is_intrusion'].sum()

        metrics['total_readings'] = total_readings
        metrics['intrusions_detected'] = intrusions_detected
        metrics['intrusion_rate'] = intrusions_detected / total_readings if total_readings > 0 else 0

        # Calculate confidence statistics
        if 'confidence_score' in detection_results.columns:
            metrics['avg_confidence'] = detection_results['confidence_score'].mean()
            metrics['confidence_std'] = detection_results['confidence_score'].std()

    return metrics

def create_time_windows(data: pd.DataFrame, window_size: str = '1H') -> pd.DataFrame:
    """Create time-based windows for analysis"""
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['time_window'] = data['timestamp'].dt.floor(window_size)
    return data

def detect_sensor_anomalies(data: pd.DataFrame, sensor_type: str = None) -> Dict[str, Any]:
    """Simple statistical anomaly detection for sensor validation"""
    if sensor_type:
        sensor_data = data[data['sensor_type'] == sensor_type]['value']
    else:
        sensor_data = data['value']

    # Calculate statistical thresholds
    mean_val = sensor_data.mean()
    std_val = sensor_data.std()

    # Define anomalies as values beyond 3 standard deviations
    lower_threshold = mean_val - 3 * std_val
    upper_threshold = mean_val + 3 * std_val

    anomalies = sensor_data[(sensor_data < lower_threshold) | (sensor_data > upper_threshold)]

    return {
        'anomaly_count': len(anomalies),
        'anomaly_percentage': len(anomalies) / len(sensor_data) * 100,
        'thresholds': {
            'lower': lower_threshold,
            'upper': upper_threshold
        },
        'statistics': {
            'mean': mean_val,
            'std': std_val,
            'min': sensor_data.min(),
            'max': sensor_data.max()
        }
    }

def merge_blockchain_with_sensor_data(blockchain_df: pd.DataFrame,
                                     sensor_df: pd.DataFrame) -> pd.DataFrame:
    """Merge blockchain data with original sensor data for analysis"""
    # Prepare blockchain data
    blockchain_df['blockchain_timestamp'] = pd.to_datetime(blockchain_df['transaction_timestamp'], unit='s')

    # Prepare sensor data
    sensor_df['sensor_timestamp'] = pd.to_datetime(sensor_df['timestamp'])

    # Merge on sensor_id and closest timestamp
    merged_data = []

    for _, sensor_row in sensor_df.iterrows():
        # Find corresponding blockchain entry
        matching_blockchain = blockchain_df[
            (blockchain_df['sensor_id'] == sensor_row['sensor_id']) &
            (blockchain_df['room'] == sensor_row['room']) &
            (blockchain_df['sensor_type'] == sensor_row['sensor_type'])
        ]

        if not matching_blockchain.empty:
            # Find closest timestamp
            time_diffs = abs(matching_blockchain['blockchain_timestamp'] - sensor_row['sensor_timestamp'])
            closest_idx = time_diffs.idxmin()
            blockchain_row = matching_blockchain.loc[closest_idx]

            # Combine data
            merged_row = sensor_row.to_dict()
            merged_row.update({
                'block_index': blockchain_row['block_index'],
                'block_hash': blockchain_row['block_hash'],
                'transaction_id': blockchain_row['transaction_id'],
                'blockchain_timestamp': blockchain_row['blockchain_timestamp'],
                'data_integrity_verified': True
            })
        else:
            # No blockchain entry found
            merged_row = sensor_row.to_dict()
            merged_row.update({
                'block_index': None,
                'block_hash': None,
                'transaction_id': None,
                'blockchain_timestamp': None,
                'data_integrity_verified': False
            })

        merged_data.append(merged_row)

    return pd.DataFrame(merged_data)

class DataIntegrityChecker:
    """Check data integrity across different components"""

    @staticmethod
    def check_blockchain_integrity(blockchain_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check blockchain data integrity"""
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }

        if 'blockchain' not in blockchain_data:
            results['is_valid'] = False
            results['errors'].append("No blockchain data found")
            return results

        blocks = blockchain_data['blockchain']

        # Check block sequence
        for i, block in enumerate(blocks):
            if block['index'] != i:
                results['errors'].append(f"Block index mismatch at position {i}")
                results['is_valid'] = False

        # Check hash chain
        for i in range(1, len(blocks)):
            if blocks[i]['previous_hash'] != blocks[i-1]['hash']:
                results['errors'].append(f"Hash chain broken at block {i}")
                results['is_valid'] = False

        return results

    @staticmethod
    def check_sensor_data_consistency(sensor_data: pd.DataFrame) -> Dict[str, Any]:
        """Check sensor data consistency"""
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }

        # Check for duplicate timestamps within same sensor
        duplicates = sensor_data.groupby(['sensor_id', 'timestamp']).size()
        duplicate_count = (duplicates > 1).sum()

        if duplicate_count > 0:
            results['warnings'].append(f"Found {duplicate_count} duplicate timestamp entries")

        # Check for temporal gaps
        sensor_data['timestamp'] = pd.to_datetime(sensor_data['timestamp'])
        for sensor_id in sensor_data['sensor_id'].unique():
            sensor_subset = sensor_data[sensor_data['sensor_id'] == sensor_id].sort_values('timestamp')
            time_diffs = sensor_subset['timestamp'].diff().dt.total_seconds()

            # Check for unusually large gaps (more than 1 hour)
            large_gaps = time_diffs > 3600
            if large_gaps.any():
                results['warnings'].append(f"Large time gaps found in sensor {sensor_id}")

        return results
