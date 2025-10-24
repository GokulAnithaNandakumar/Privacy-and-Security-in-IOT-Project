import numpy as np
import pandas as pd
from typing import Union, List, Dict
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import json

class DifferentialPrivacy:
    """
    Implements differential privacy techniques for IoT sensor data
    """

    def __init__(self, epsilon: float = 1.0, sensitivity: float = 1.0):
        """
        Initialize differential privacy parameters

        Args:
            epsilon: Privacy parameter (smaller = more private)
            sensitivity: Maximum change in output for single record change
        """
        self.epsilon = epsilon
        self.sensitivity = sensitivity

    def laplace_noise(self, size: Union[int, tuple] = 1) -> Union[float, np.ndarray]:
        """Generate Laplace noise for differential privacy"""
        scale = self.sensitivity / self.epsilon
        return np.random.laplace(0, scale, size)

    def gaussian_noise(self, size: Union[int, tuple] = 1, delta: float = 1e-5) -> Union[float, np.ndarray]:
        """Generate Gaussian noise for differential privacy"""
        # For (epsilon, delta)-differential privacy
        scale = np.sqrt(2 * np.log(1.25 / delta)) * self.sensitivity / self.epsilon
        return np.random.normal(0, scale, size)

    def add_noise_to_value(self, value: float, noise_type: str = 'laplace') -> float:
        """Add privacy-preserving noise to a single value"""
        if noise_type == 'laplace':
            noise = self.laplace_noise()
        elif noise_type == 'gaussian':
            noise = self.gaussian_noise()
        else:
            raise ValueError("noise_type must be 'laplace' or 'gaussian'")

        return value + noise

    def add_noise_to_series(self, series: pd.Series, noise_type: str = 'laplace') -> pd.Series:
        """Add privacy-preserving noise to a pandas Series"""
        if noise_type == 'laplace':
            noise = self.laplace_noise(len(series))
        elif noise_type == 'gaussian':
            noise = self.gaussian_noise(len(series))
        else:
            raise ValueError("noise_type must be 'laplace' or 'gaussian'")

        return series + noise

    def k_anonymity_grouping(self, data: pd.DataFrame, k: int = 5,
                           sensitive_columns: List[str] = None) -> pd.DataFrame:
        """
        Implement k-anonymity by grouping similar records

        Args:
            data: Input dataframe
            k: Minimum group size
            sensitive_columns: Columns to anonymize
        """
        if sensitive_columns is None:
            sensitive_columns = ['value']

        anonymized_data = data.copy()

        # Group by room and sensor_type for k-anonymity
        grouped = anonymized_data.groupby(['room', 'sensor_type'])

        result_groups = []
        for name, group in grouped:
            if len(group) >= k:
                # Apply aggregation to ensure k-anonymity
                for col in sensitive_columns:
                    if col in group.columns:
                        # Replace individual values with group mean + noise
                        group_mean = group[col].mean()
                        noise = self.laplace_noise(len(group))
                        group[col] = group_mean + noise

                result_groups.append(group)
            else:
                # If group is too small, add more noise for privacy
                for col in sensitive_columns:
                    if col in group.columns:
                        large_noise = self.laplace_noise(len(group)) * 2
                        group[col] = group[col] + large_noise

                result_groups.append(group)

        if result_groups:
            return pd.concat(result_groups, ignore_index=True)
        else:
            return anonymized_data

    def temporal_aggregation(self, data: pd.DataFrame, window: str = '1H') -> pd.DataFrame:
        """
        Aggregate data over time windows for privacy

        Args:
            data: Input dataframe with timestamp column
            window: Time window for aggregation (e.g., '1H', '30min')
        """
        data['timestamp'] = pd.to_datetime(data['timestamp'])

        # Group by time windows and other categorical columns
        aggregated_groups = []

        for (room, sensor_type), group in data.groupby(['room', 'sensor_type']):
            # Resample by time window
            group_resampled = group.set_index('timestamp').resample(window).agg({
                'value': ['mean', 'std', 'count'],
                'sensor_id': 'first'
            })

            # Flatten column names
            group_resampled.columns = ['_'.join(col).strip() for col in group_resampled.columns]
            group_resampled = group_resampled.reset_index()

            # Add back categorical columns
            group_resampled['room'] = room
            group_resampled['sensor_type'] = sensor_type

            # Add noise to aggregated values
            if 'value_mean' in group_resampled.columns:
                noise = self.laplace_noise(len(group_resampled))
                group_resampled['value_mean'] += noise

            aggregated_groups.append(group_resampled)

        if aggregated_groups:
            return pd.concat(aggregated_groups, ignore_index=True)
        else:
            return pd.DataFrame()


class DataEncryption:
    """
    Handles encryption and decryption of IoT sensor data
    """

    def __init__(self, password: str = None):
        """Initialize encryption with password-based key derivation"""
        if password is None:
            password = "iot_security_demo_2024"

        self.password = password.encode()
        self.key = self._derive_key()
        self.cipher = Fernet(self.key)

    def _derive_key(self) -> bytes:
        """Derive encryption key from password"""
        salt = b'iot_salt_2024'  # In production, use random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.password))
        return key

    def encrypt_value(self, value: Union[str, float, int]) -> str:
        """Encrypt a single value"""
        value_str = str(value)
        encrypted_bytes = self.cipher.encrypt(value_str.encode())
        return base64.urlsafe_b64encode(encrypted_bytes).decode()

    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a single value"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted_bytes = self.cipher.decrypt(encrypted_bytes)
            return decrypted_bytes.decode()
        except Exception as e:
            return f"Decryption failed: {str(e)}"

    def encrypt_dataframe(self, data: pd.DataFrame,
                         columns_to_encrypt: List[str] = None) -> pd.DataFrame:
        """Encrypt specified columns in a dataframe"""
        if columns_to_encrypt is None:
            columns_to_encrypt = ['value']

        encrypted_data = data.copy()

        for col in columns_to_encrypt:
            if col in encrypted_data.columns:
                encrypted_data[f"{col}_encrypted"] = encrypted_data[col].apply(
                    self.encrypt_value
                )
                # Optionally remove original column
                # encrypted_data = encrypted_data.drop(columns=[col])

        return encrypted_data

    def decrypt_dataframe(self, data: pd.DataFrame,
                         columns_to_decrypt: List[str] = None) -> pd.DataFrame:
        """Decrypt specified columns in a dataframe"""
        if columns_to_decrypt is None:
            columns_to_decrypt = [col for col in data.columns if col.endswith('_encrypted')]

        decrypted_data = data.copy()

        for col in columns_to_decrypt:
            if col in decrypted_data.columns:
                original_col = col.replace('_encrypted', '')
                decrypted_data[f"{original_col}_decrypted"] = decrypted_data[col].apply(
                    self.decrypt_value
                )

        return decrypted_data


class PrivacyPreservationPipeline:
    """
    Complete privacy preservation pipeline for IoT data
    """

    def __init__(self, epsilon: float = 1.0, k_value: int = 5, encrypt: bool = True):
        """
        Initialize privacy preservation pipeline

        Args:
            epsilon: Differential privacy parameter
            k_value: k-anonymity parameter
            encrypt: Whether to apply encryption
        """
        self.dp = DifferentialPrivacy(epsilon=epsilon)
        self.encryption = DataEncryption() if encrypt else None
        self.k_value = k_value
        self.encrypt = encrypt

    def apply_privacy_techniques(self, data: pd.DataFrame,
                               techniques: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Apply multiple privacy techniques to the data

        Args:
            data: Input sensor data
            techniques: List of techniques to apply
                       ['differential_privacy', 'k_anonymity', 'encryption', 'aggregation']
        """
        if techniques is None:
            techniques = ['differential_privacy', 'k_anonymity', 'encryption']

        results = {'original': data.copy()}

        if 'differential_privacy' in techniques:
            dp_data = data.copy()
            dp_data['value'] = self.dp.add_noise_to_series(dp_data['value'])
            results['differential_privacy'] = dp_data

        if 'k_anonymity' in techniques:
            k_anon_data = self.dp.k_anonymity_grouping(data, k=self.k_value)
            results['k_anonymity'] = k_anon_data

        if 'encryption' in techniques and self.encryption:
            encrypted_data = self.encryption.encrypt_dataframe(data)
            results['encryption'] = encrypted_data

        if 'aggregation' in techniques:
            agg_data = self.dp.temporal_aggregation(data, window='1H')
            results['aggregation'] = agg_data

        if 'combined' in techniques:
            # Apply multiple techniques in sequence
            combined_data = data.copy()

            # 1. Add differential privacy noise
            combined_data['value'] = self.dp.add_noise_to_series(combined_data['value'])

            # 2. Apply k-anonymity
            combined_data = self.dp.k_anonymity_grouping(combined_data, k=self.k_value)

            # 3. Encrypt if requested
            if self.encryption:
                combined_data = self.encryption.encrypt_dataframe(combined_data)

            results['combined'] = combined_data

        return results

    def evaluate_privacy_utility_tradeoff(self, original_data: pd.DataFrame,
                                        private_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the utility loss due to privacy preservation
        """
        # Calculate various utility metrics
        metrics = {}

        if 'value' in original_data.columns and 'value' in private_data.columns:
            # Mean absolute error
            mae = np.mean(np.abs(original_data['value'] - private_data['value']))
            metrics['mean_absolute_error'] = mae

            # Correlation preservation
            if len(original_data) > 1 and len(private_data) > 1:
                correlation = np.corrcoef(original_data['value'], private_data['value'])[0, 1]
                metrics['correlation'] = correlation if not np.isnan(correlation) else 0.0

            # Signal-to-noise ratio
            signal_power = np.var(original_data['value'])
            noise_power = np.var(original_data['value'] - private_data['value'])
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                metrics['signal_to_noise_ratio'] = snr

        return metrics


if __name__ == "__main__":
    # Example usage
    from ..sensors.sensor_simulator import IoTSensorSimulator

    # Generate sample data
    simulator = IoTSensorSimulator(num_rooms=2)
    sample_data = simulator.generate_full_dataset(duration_hours=24)

    # Apply privacy preservation
    privacy_pipeline = PrivacyPreservationPipeline(epsilon=1.0, k_value=3)
    privacy_results = privacy_pipeline.apply_privacy_techniques(
        sample_data,
        techniques=['differential_privacy', 'k_anonymity', 'encryption', 'combined']
    )

    # Evaluate privacy-utility tradeoff
    metrics = privacy_pipeline.evaluate_privacy_utility_tradeoff(
        privacy_results['original'],
        privacy_results['differential_privacy']
    )

    print("Privacy preservation applied successfully!")
    print(f"Available techniques: {list(privacy_results.keys())}")
    print(f"Utility metrics: {metrics}")
