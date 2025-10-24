"""
Real Blockchain IoT Data Manager
================================
Connects to Ethereum blockchain to store IoT sensor data securely
Implements privacy preservation and ML intrusion detection
"""

import os
import json
import hashlib
import logging
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from web3 import Web3
from eth_account import Account
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import numpy as np
from sklearn.ensemble import IsolationForest
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SensorReading:
    """Structure for IoT sensor readings"""
    sensor_id: str
    room: str
    sensor_type: str
    value: float
    timestamp: float
    unit: str

@dataclass
class PrivacyConfig:
    """Privacy preservation configuration"""
    epsilon: float = 1.0  # Differential privacy epsilon
    k_anonymity: int = 3  # K-anonymity level
    encryption_key: bytes = None

class RealBlockchainIoTManager:
    """
    Real blockchain implementation for IoT data storage
    Uses Ethereum blockchain with smart contracts
    """

    def __init__(self,
                 blockchain_url: str = "http://127.0.0.1:8545",  # Local Ganache
                 private_key: str = None,
                 contract_address: str = None):
        """
        Initialize blockchain connection

        Args:
            blockchain_url: Ethereum node URL (Ganache, Infura, etc.)
            private_key: Private key for transaction signing
            contract_address: Deployed smart contract address
        """
        self.w3 = Web3(Web3.HTTPProvider(blockchain_url))
        self.private_key = private_key
        self.contract_address = contract_address

        # Privacy configuration
        self.privacy_config = PrivacyConfig()
        self.privacy_config.encryption_key = get_random_bytes(32)  # AES-256 key

        # ML model for intrusion detection
        self.ml_model = IsolationForest(
            contamination=0.1,  # 10% expected anomalies
            random_state=42,
            n_estimators=100
        )
        self.ml_trained = False
        self.sensor_data_buffer = []

        # File storage tracking
        self.stored_files = []

        # Load account from private key
        if private_key:
            self.account = Account.from_key(private_key)
            logger.info(f"Account loaded: {self.account.address}")

        # Load smart contract
        self.contract = None
        if contract_address:
            self.load_contract(contract_address)

        logger.info("Real Blockchain IoT Manager initialized")

    def load_contract(self, contract_address: str):
        """Load the deployed smart contract"""
        try:
            # Load contract ABI (you would need to compile and get this)
            with open('contracts/IoTDataStorage_abi.json', 'r') as f:
                contract_abi = json.load(f)

            self.contract = self.w3.eth.contract(
                address=contract_address,
                abi=contract_abi
            )
            self.contract_address = contract_address
            logger.info(f"Contract loaded at: {contract_address}")

        except FileNotFoundError:
            logger.warning("Contract ABI not found. You need to compile the contract first.")
            # For demo purposes, create a mock ABI
            self.create_mock_contract_interface()

    def create_mock_contract_interface(self):
        """Create a mock interface for demonstration"""
        logger.info("Creating mock contract interface for demo")
        # This would be replaced with actual deployed contract
        self.contract_address = "0x" + "0" * 40  # Mock address

    def apply_differential_privacy(self, value: float, sensitivity: float = 1.0) -> float:
        """
        Apply differential privacy to sensor value

        Args:
            value: Original sensor value
            sensitivity: Global sensitivity of the function

        Returns:
            Privacy-preserved value with noise added
        """
        # Laplace mechanism for ε-differential privacy
        scale = sensitivity / self.privacy_config.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise

    def apply_k_anonymity(self, sensor_data: List[SensorReading]) -> List[SensorReading]:
        """
        Apply k-anonymity by grouping similar sensor readings

        Args:
            sensor_data: List of sensor readings

        Returns:
            K-anonymized sensor data
        """
        if len(sensor_data) < self.privacy_config.k_anonymity:
            return sensor_data

        # Group by sensor type and room for k-anonymity
        df = pd.DataFrame([
            {
                'sensor_id': reading.sensor_id,
                'room': reading.room,
                'sensor_type': reading.sensor_type,
                'value': reading.value,
                'timestamp': reading.timestamp,
                'unit': reading.unit
            }
            for reading in sensor_data
        ])

        # Generalize values to achieve k-anonymity
        grouped = df.groupby(['room', 'sensor_type'])
        anonymized_data = []

        for (room, sensor_type), group in grouped:
            if len(group) >= self.privacy_config.k_anonymity:
                # Use average value for k-anonymity
                avg_value = group['value'].mean()
                for _, row in group.iterrows():
                    anonymized_data.append(SensorReading(
                        sensor_id=f"{room}_{sensor_type}_anonymized",
                        room=room,
                        sensor_type=sensor_type,
                        value=avg_value,
                        timestamp=row['timestamp'],
                        unit=row['unit']
                    ))
            else:
                # Keep original if group too small
                for _, row in group.iterrows():
                    anonymized_data.append(SensorReading(
                        sensor_id=row['sensor_id'],
                        room=row['room'],
                        sensor_type=row['sensor_type'],
                        value=row['value'],
                        timestamp=row['timestamp'],
                        unit=row['unit']
                    ))

        return anonymized_data

    def encrypt_data(self, data: str) -> Tuple[bytes, bytes]:
        """
        Encrypt sensor data using AES-256-CBC

        Args:
            data: Data to encrypt

        Returns:
            Tuple of (encrypted_data, iv)
        """
        # Generate random IV
        iv = get_random_bytes(AES.block_size)

        # Create cipher
        cipher = AES.new(self.privacy_config.encryption_key, AES.MODE_CBC, iv)

        # Pad and encrypt data
        padded_data = pad(data.encode('utf-8'), AES.block_size)
        encrypted_data = cipher.encrypt(padded_data)

        return encrypted_data, iv

    def decrypt_data(self, encrypted_data: bytes, iv: bytes) -> str:
        """
        Decrypt sensor data

        Args:
            encrypted_data: Encrypted data
            iv: Initialization vector

        Returns:
            Decrypted data string
        """
        cipher = AES.new(self.privacy_config.encryption_key, AES.MODE_CBC, iv)
        decrypted_data = unpad(cipher.decrypt(encrypted_data), AES.block_size)
        return decrypted_data.decode('utf-8')

    def train_ml_model(self, sensor_data: List[SensorReading]):
        """
        Train ML model for intrusion detection

        Args:
            sensor_data: Historical sensor data for training

        Returns:
            bool: True if training successful, False otherwise
        """
        if len(sensor_data) < 10:  # Need minimum data for training
            logger.warning("Not enough data for ML training")
            return False

        try:
            # Extract features for ML
            features = []
            for reading in sensor_data:
                # Create feature vector: [hour, day_of_week, value, sensor_type_encoded]
                dt = datetime.fromtimestamp(reading.timestamp)
                sensor_type_encoded = hash(reading.sensor_type) % 1000  # Simple encoding

                features.append([
                    dt.hour,
                    dt.weekday(),
                    reading.value,
                    sensor_type_encoded
                ])

            X = np.array(features)

            # Train isolation forest
            self.ml_model.fit(X)
            self.ml_trained = True

            logger.info(f"ML model trained on {len(sensor_data)} sensor readings")
            return True

        except Exception as e:
            logger.error(f"Error training ML model: {e}")
            return False

    def detect_intrusion(self, reading: SensorReading) -> Tuple[bool, float]:
        """
        Detect if sensor reading is anomalous using ML

        Args:
            reading: Sensor reading to analyze

        Returns:
            Tuple of (is_anomalous, confidence_score)
        """
        if not self.ml_trained:
            return False, 0.0

        # Extract features
        dt = datetime.fromtimestamp(reading.timestamp)
        sensor_type_encoded = hash(reading.sensor_type) % 1000

        features = np.array([[
            dt.hour,
            dt.weekday(),
            reading.value,
            sensor_type_encoded
        ]])

        # Predict anomaly
        prediction = self.ml_model.predict(features)[0]
        anomaly_score = self.ml_model.decision_function(features)[0]

        is_anomalous = prediction == -1
        confidence_score = abs(anomaly_score) * 100  # Convert to percentage

        return is_anomalous, min(confidence_score, 100.0)

    def create_data_hash(self, reading: SensorReading) -> str:
        """
        Create hash of sensor data for integrity verification

        Args:
            reading: Sensor reading

        Returns:
            SHA-256 hash of the data
        """
        data_string = f"{reading.sensor_id}{reading.room}{reading.sensor_type}{reading.value}{reading.timestamp}"
        return hashlib.sha256(data_string.encode()).hexdigest()

    def store_on_blockchain(self, reading: SensorReading) -> Optional[str]:
        """
        Store sensor data on blockchain with privacy preservation

        Args:
            reading: Sensor reading to store

        Returns:
            Transaction hash if successful
        """
        try:
            # Apply privacy preservation
            dp_value = self.apply_differential_privacy(reading.value)

            # Encrypt the privacy-preserved value
            encrypted_data, iv = self.encrypt_data(str(dp_value))

            # Create data hash for integrity
            data_hash = self.create_data_hash(reading)

            # Detect intrusions
            is_anomalous, confidence = self.detect_intrusion(reading)

            if self.contract and self.private_key:
                # Prepare transaction
                nonce = self.w3.eth.get_transaction_count(self.account.address)

                # Build transaction
                transaction = self.contract.functions.recordSensorData(
                    reading.sensor_id,
                    reading.room,
                    reading.sensor_type,
                    Web3.keccak(encrypted_data),  # Store hash of encrypted data
                    Web3.keccak(text=data_hash),  # Store hash for integrity
                    is_anomalous,
                    int(confidence)
                ).build_transaction({
                    'from': self.account.address,
                    'nonce': nonce,
                    'gas': 500000,
                    'gasPrice': self.w3.to_wei('20', 'gwei')
                })

                # Sign and send transaction
                signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)

                # Wait for confirmation
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

                logger.info(f"Data stored on blockchain. TX: {receipt.transactionHash.hex()}")

                if is_anomalous:
                    logger.warning(f"🚨 INTRUSION DETECTED: {reading.sensor_id} - Confidence: {confidence:.1f}%")

                return receipt.transactionHash.hex()

            else:
                # Demo mode - simulate blockchain storage
                tx_hash = f"0x{hashlib.sha256(f'{reading.sensor_id}{reading.timestamp}'.encode()).hexdigest()}"

                logger.info(f"[DEMO] Data stored on blockchain. TX: {tx_hash}")

                if is_anomalous:
                    logger.warning(f"🚨 INTRUSION DETECTED: {reading.sensor_id} - Confidence: {confidence:.1f}%")

                return tx_hash

        except Exception as e:
            logger.error(f"Failed to store data on blockchain: {e}")
            return None

    def process_sensor_reading(self, sensor_id: str, room: str, sensor_type: str,
                             value: float, unit: str = "unknown") -> Optional[str]:
        """
        Process a new sensor reading with full privacy and security pipeline

        Args:
            sensor_id: Unique sensor identifier
            room: Room location
            sensor_type: Type of sensor
            value: Sensor value
            unit: Unit of measurement

        Returns:
            Transaction hash if successful
        """
        # Create sensor reading
        reading = SensorReading(
            sensor_id=sensor_id,
            room=room,
            sensor_type=sensor_type,
            value=value,
            timestamp=datetime.now().timestamp(),
            unit=unit
        )

        # Add to buffer for ML training
        self.sensor_data_buffer.append(reading)

        # Automatic ML model training when sufficient data is available
        if len(self.sensor_data_buffer) >= 10 and not self.ml_trained:
            logger.info("Auto-training ML model with sufficient data...")
            self.train_ml_model(self.sensor_data_buffer)

        # Store on blockchain
        return self.store_on_blockchain(reading)

    def get_blockchain_stats(self) -> Dict:
        """Get blockchain contract statistics"""
        try:
            if self.contract:
                stats = self.contract.functions.getContractStats().call()
                return {
                    'total_records': stats[0],
                    'anomalous_records': stats[1],
                    'epsilon': stats[2] / 100.0,  # Convert back from scaled value
                    'k_anonymity': stats[3],
                    'encryption': stats[4]
                }
            else:
                # Demo mode stats
                return {
                    'total_records': len(self.sensor_data_buffer),
                    'anomalous_records': sum(1 for r in self.sensor_data_buffer
                                           if self.detect_intrusion(r)[0]),
                    'epsilon': self.privacy_config.epsilon,
                    'k_anonymity': self.privacy_config.k_anonymity,
                    'encryption': 'AES-256-CBC'
                }
        except Exception as e:
            logger.error(f"Failed to get blockchain stats: {e}")
            return {}

    def get_recent_anomalies(self, limit: int = 10) -> List[Dict]:
        """Get recent anomalous sensor readings"""
        anomalies = []

        for reading in reversed(self.sensor_data_buffer[-100:]):  # Check last 100
            is_anomalous, confidence = self.detect_intrusion(reading)
            if is_anomalous:
                anomalies.append({
                    'sensor_id': reading.sensor_id,
                    'room': reading.room,
                    'sensor_type': reading.sensor_type,
                    'value': reading.value,
                    'timestamp': reading.timestamp,
                    'confidence': confidence
                })

                if len(anomalies) >= limit:
                    break

        return anomalies

    def train_ml_model_manual(self) -> Dict:
        """
        Manually train ML model with current sensor data

        Returns:
            Training results with statistics
        """
        try:
            if len(self.sensor_data_buffer) < 10:
                return {
                    'success': False,
                    'message': f'Need at least 10 data points. Currently have {len(self.sensor_data_buffer)}',
                    'data_points': len(self.sensor_data_buffer)
                }

            # Prepare training data (use 80% for training)
            training_size = int(len(self.sensor_data_buffer) * 0.8)
            training_data = self.sensor_data_buffer[:training_size]

            # Train the model
            training_result = self.train_ml_model(training_data)

            if training_result:
                # Test on remaining data
                test_data = self.sensor_data_buffer[training_size:]
                anomalies_detected = 0

                for reading in test_data:
                    is_anomalous, confidence = self.detect_intrusion(reading)
                    if is_anomalous:
                        anomalies_detected += 1

                return {
                    'success': True,
                    'message': 'ML model trained successfully',
                    'training_data_points': len(training_data),
                    'test_data_points': len(test_data),
                    'anomalies_detected': anomalies_detected,
                    'anomaly_rate': (anomalies_detected / len(test_data) * 100) if test_data else 0,
                    'model_trained': self.ml_trained
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to train ML model',
                    'data_points': len(self.sensor_data_buffer)
                }

        except Exception as e:
            logger.error(f"Error in manual ML training: {e}")
            return {
                'success': False,
                'message': f'Training error: {str(e)}',
                'data_points': len(self.sensor_data_buffer)
            }

    def get_stored_files(self) -> List[Dict]:
        """Get list of all stored files with metadata"""
        # Add demo files for demonstration
        demo_files = [
            {
                'name': 'sensor_config.json',
                'size': 2340,
                'type': 'application/json',
                'tx_hash': '0x1234567890abcdef1234567890abcdef12345678',
                'public': False,
                'timestamp': time.time() - 3600  # 1 hour ago
            },
            {
                'name': 'privacy_policy.pdf',
                'size': 159744,
                'type': 'application/pdf',
                'tx_hash': '0x234567890abcdef1234567890abcdef123456789',
                'public': True,
                'timestamp': time.time() - 7200  # 2 hours ago
            }
        ]

        # Combine demo files with actually stored files
        all_files = demo_files + self.stored_files

        # Sort by timestamp (newest first)
        all_files.sort(key=lambda x: x['timestamp'], reverse=True)

        return all_files

    def store_file_on_blockchain(self, file_name: str, file_content: bytes,
                                content_type: str = "application/octet-stream",
                                is_public: bool = False) -> str:
        """
        Store file on blockchain with encryption and privacy preservation

        Args:
            file_name: Name of the file
            file_content: Raw file content
            content_type: MIME type of the file
            is_public: Whether file should be publicly accessible

        Returns:
            Transaction hash or demo ID
        """
        try:
            # Calculate file hash
            file_hash = Web3.keccak(file_content)

            # Encrypt file content
            cipher = AES.new(self.privacy_config.encryption_key, AES.MODE_CBC)

            # Pad content to AES block size
            pad_length = AES.block_size - (len(file_content) % AES.block_size)
            padded_content = file_content + bytes([pad_length]) * pad_length

            encrypted_content = cipher.encrypt(padded_content)
            iv = cipher.iv

            # Store IV with encrypted content
            full_encrypted_content = iv + encrypted_content

            if self.contract and self.account:
                # Real blockchain storage
                function_call = self.contract.functions.storeFile(
                    file_name,
                    file_hash,
                    full_encrypted_content[:1024],  # Limit for on-chain storage
                    "",  # IPFS hash (empty for now)
                    len(file_content),
                    content_type,
                    is_public
                )

                transaction = function_call.build_transaction({
                    'from': self.account.address,
                    'nonce': self.w3.eth.get_transaction_count(self.account.address),
                    'gas': 500000,
                    'gasPrice': self.w3.to_wei('10', 'gwei')
                })

                signed_tx = self.w3.eth.account.sign_transaction(transaction, self.private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

                # Track the stored file
                file_record = {
                    'name': file_name,
                    'size': len(file_content),
                    'type': content_type,
                    'tx_hash': tx_hash.hex(),
                    'public': is_public,
                    'timestamp': time.time()
                }
                self.stored_files.append(file_record)

                logger.info(f"📁 File stored on blockchain: {tx_hash.hex()}")
                return tx_hash.hex()

            else:
                # Demo mode - simulate storage
                demo_tx_hash = f"demo_file_{len(file_content)}_{int(time.time())}"

                # Track the stored file in demo mode
                file_record = {
                    'name': file_name,
                    'size': len(file_content),
                    'type': content_type,
                    'tx_hash': demo_tx_hash,
                    'public': is_public,
                    'timestamp': time.time()
                }
                self.stored_files.append(file_record)

                logger.info(f"📁 [DEMO] File stored: {demo_tx_hash}")
                return demo_tx_hash

        except Exception as e:
            logger.error(f"Failed to store file on blockchain: {e}")
            return f"error_{int(time.time())}"

    def get_all_transactions(self) -> List[Dict]:
        """
        Get all blockchain transactions for visualization

        Returns:
            List of transaction details
        """
        transactions = []

        if self.contract and self.account:
            try:
                # Get all sensor data events
                sensor_events = self.contract.events.SensorDataRecorded.get_logs(fromBlock=0)
                for event in sensor_events:
                    transactions.append({
                        'type': 'Sensor Data',
                        'tx_hash': event['transactionHash'].hex(),
                        'block_number': event['blockNumber'],
                        'timestamp': datetime.fromtimestamp(event['args']['timestamp']),
                        'sensor_id': event['args']['sensorId'],
                        'room': event['args']['room'],
                        'is_anomalous': event['args']['isAnomalous'],
                        'gas_used': None  # Would need to fetch transaction receipt
                    })

                # Get all file storage events
                file_events = self.contract.events.FileStored.get_logs(fromBlock=0)
                for event in file_events:
                    transactions.append({
                        'type': 'File Storage',
                        'tx_hash': event['transactionHash'].hex(),
                        'block_number': event['blockNumber'],
                        'timestamp': datetime.fromtimestamp(event['args']['timestamp']),
                        'file_name': event['args']['fileName'],
                        'file_size': event['args']['fileSize'],
                        'uploader': event['args']['uploader'],
                        'gas_used': None
                    })

                # Get intrusion detection events
                intrusion_events = self.contract.events.IntrusionDetected.get_logs(fromBlock=0)
                for event in intrusion_events:
                    transactions.append({
                        'type': 'Intrusion Alert',
                        'tx_hash': event['transactionHash'].hex(),
                        'block_number': event['blockNumber'],
                        'timestamp': datetime.fromtimestamp(event['args']['timestamp']),
                        'sensor_id': event['args']['sensorId'],
                        'confidence': event['args']['confidenceScore'],
                        'gas_used': None
                    })

            except Exception as e:
                logger.error(f"Failed to fetch blockchain events: {e}")

        else:
            # Demo mode - simulate transactions
            for i, reading in enumerate(self.sensor_data_buffer[-20:]):  # Last 20
                is_anomalous, confidence = self.detect_intrusion(reading)
                transactions.append({
                    'type': 'Sensor Data',
                    'tx_hash': f"demo_tx_{i}_{int(reading.timestamp)}",
                    'block_number': i + 1,
                    'timestamp': datetime.fromtimestamp(reading.timestamp),
                    'sensor_id': reading.sensor_id,
                    'room': reading.room,
                    'is_anomalous': is_anomalous,
                    'gas_used': 21000 + (5000 if is_anomalous else 0)
                })

                if is_anomalous:
                    transactions.append({
                        'type': 'Intrusion Alert',
                        'tx_hash': f"demo_alert_{i}_{int(reading.timestamp)}",
                        'block_number': i + 1,
                        'timestamp': datetime.fromtimestamp(reading.timestamp),
                        'sensor_id': reading.sensor_id,
                        'confidence': confidence,
                        'gas_used': 15000
                    })

        # Sort by timestamp (newest first)
        transactions.sort(key=lambda x: x['timestamp'], reverse=True)
        return transactions

    def get_blockchain_visualization_data(self) -> Dict:
        """
        Get data for blockchain visualization

        Returns:
            Comprehensive blockchain analytics data
        """
        transactions = self.get_all_transactions()

        # Transaction type distribution
        type_counts = {}
        for tx in transactions:
            tx_type = tx['type']
            type_counts[tx_type] = type_counts.get(tx_type, 0) + 1

        # Transactions over time
        tx_timeline = []
        for tx in transactions:
            tx_timeline.append({
                'timestamp': tx['timestamp'],
                'type': tx['type'],
                'count': 1
            })

        # Room activity (for sensor data)
        room_activity = {}
        for tx in transactions:
            if tx['type'] == 'Sensor Data' and 'room' in tx:
                room = tx['room']
                room_activity[room] = room_activity.get(room, 0) + 1

        # Anomaly detection stats
        total_sensor_tx = sum(1 for tx in transactions if tx['type'] == 'Sensor Data')
        anomalous_tx = sum(1 for tx in transactions
                          if tx['type'] == 'Sensor Data' and tx.get('is_anomalous', False))

        return {
            'transactions': transactions,
            'type_distribution': type_counts,
            'timeline': tx_timeline,
            'room_activity': room_activity,
            'security_stats': {
                'total_sensor_transactions': total_sensor_tx,
                'anomalous_transactions': anomalous_tx,
                'security_ratio': (anomalous_tx / total_sensor_tx * 100) if total_sensor_tx > 0 else 0,
                'intrusion_alerts': sum(1 for tx in transactions if tx['type'] == 'Intrusion Alert')
            },
            'network_stats': {
                'total_transactions': len(transactions),
                'avg_gas_used': sum(tx.get('gas_used', 21000) for tx in transactions) / len(transactions) if transactions else 0,
                'latest_block': max(tx['block_number'] for tx in transactions) if transactions else 0
            }
        }