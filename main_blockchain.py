"""
IoT-Based Privacy-Preserving Data Management with Blockchain Integration and ML Intrusion Detection
===================================================================================================

Main application for real blockchain-based IoT privacy and security system
Course: Privacy and Security in IoT

Features:
- Real Ethereum blockchain integration with smart contracts
- Privacy-preserving techniques (Differential Privacy, K-Anonymity, AES Encryption)
- Machine Learning-based intrusion detection (Isolation Forest)
- MQTT IoT communication protocol
- Real-time monitoring dashboard
"""

import os
import time
import json
import threading
import logging
from datetime import datetime
from typing import Dict, List
from dataclasses import asdict

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from blockchain_iot_manager import RealBlockchainIoTManager, SensorReading
from src.sensors.sensor_simulator import IoTSensorSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IoTPrivacySecuritySystem:
    """
    Main IoT Privacy and Security System with Real Blockchain Integration

    This system demonstrates:
    1. Real blockchain storage using Ethereum smart contracts
    2. Privacy preservation with differential privacy and k-anonymity
    3. ML-based intrusion detection using Isolation Forest
    4. MQTT communication for IoT devices
    5. Real-time monitoring and alerting
    """

    def __init__(self, config_file: str = "config.json"):
        """Initialize the IoT Privacy & Security System"""

        # Load configuration
        self.config = self.load_config(config_file)

        # Initialize blockchain manager
        self.blockchain_manager = RealBlockchainIoTManager(
            blockchain_url=self.config.get('blockchain_url', 'http://127.0.0.1:8545'),
            private_key=self.config.get('private_key'),
            contract_address=self.config.get('contract_address')
        )

        # Initialize IoT sensor simulator
        self.sensor_simulator = IoTSensorSimulator(
            num_rooms=self.config.get('num_rooms', 4)
        )

        # System state
        self.is_running = False
        self.sensor_thread = None
        self.processed_data = []
        self.alerts = []

        logger.info("IoT Privacy & Security System initialized")

    def load_config(self, config_file: str) -> Dict:
        """Load system configuration"""
        default_config = {
            'blockchain_url': 'http://127.0.0.1:8545',  # Local Ganache
            'private_key': None,  # Will use demo mode if not provided
            'contract_address': None,  # Will use demo mode if not provided
            'num_rooms': 4,
            'sensors_per_room': 5,
            'data_collection_interval': 30,  # seconds
            'privacy_epsilon': 1.0,
            'k_anonymity_level': 3,
            'ml_training_threshold': 50
        }

        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Configuration loaded from {config_file}")
            else:
                logger.info("Using default configuration")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")

        return default_config

    def start_system(self):
        """Start the IoT privacy and security system"""
        if self.is_running:
            logger.warning("System is already running")
            return

        self.is_running = True
        logger.info("🚀 Starting IoT Privacy & Security System...")

        # Start sensor data collection in background
        self.sensor_thread = threading.Thread(target=self._sensor_data_loop, daemon=True)
        self.sensor_thread.start()

        logger.info("✅ System started successfully")

    def stop_system(self):
        """Stop the system"""
        self.is_running = False
        logger.info("🛑 Stopping IoT Privacy & Security System...")

        if self.sensor_thread:
            self.sensor_thread.join(timeout=5)

        logger.info("✅ System stopped")

    def _sensor_data_loop(self):
        """Background loop for collecting and processing sensor data"""
        logger.info("📡 Starting sensor data collection loop...")

        while self.is_running:
            try:
                # Generate sensor readings from all rooms
                rooms = ['living_room', 'bedroom', 'kitchen', 'office']
                sensor_types = ['temperature', 'humidity', 'motion', 'light', 'air_quality']

                for room in rooms:
                    for sensor_type in sensor_types:
                        # Generate realistic sensor data
                        value = self._generate_realistic_sensor_value(sensor_type, room)
                        sensor_id = f"{room}_{sensor_type}_sensor_001"
                        unit = self._get_sensor_unit(sensor_type)

                        # Process through privacy-preserving blockchain pipeline
                        tx_hash = self.blockchain_manager.process_sensor_reading(
                            sensor_id=sensor_id,
                            room=room,
                            sensor_type=sensor_type,
                            value=value,
                            unit=unit
                        )

                        if tx_hash:
                            # Store processed data for analytics
                            self.processed_data.append({
                                'timestamp': datetime.now(),
                                'sensor_id': sensor_id,
                                'room': room,
                                'sensor_type': sensor_type,
                                'value': value,
                                'unit': unit,
                                'tx_hash': tx_hash
                            })

                            # Check for anomalies and create alerts
                            reading = SensorReading(sensor_id, room, sensor_type, value, time.time(), unit)
                            is_anomalous, confidence = self.blockchain_manager.detect_intrusion(reading)

                            if is_anomalous:
                                alert = {
                                    'timestamp': datetime.now(),
                                    'sensor_id': sensor_id,
                                    'room': room,
                                    'sensor_type': sensor_type,
                                    'value': value,
                                    'confidence': confidence,
                                    'severity': 'HIGH' if confidence > 80 else 'MEDIUM'
                                }
                                self.alerts.append(alert)
                                logger.warning(f"🚨 INTRUSION ALERT: {sensor_id} - {confidence:.1f}% confidence")

                # Wait before next collection cycle
                time.sleep(self.config['data_collection_interval'])

            except Exception as e:
                logger.error(f"Error in sensor data loop: {e}")
                time.sleep(5)  # Wait before retrying

    def _generate_realistic_sensor_value(self, sensor_type: str, room: str) -> float:
        """Generate realistic sensor values based on type and context"""
        import random

        # Base values influenced by room type and time
        hour = datetime.now().hour

        if sensor_type == 'temperature':
            base_temp = 22.0  # Base room temperature
            if room == 'kitchen':
                base_temp += 2.0  # Kitchen warmer due to appliances
            elif room == 'bedroom':
                base_temp -= 1.0  # Bedroom slightly cooler

            # Add day/night variation
            if 6 <= hour <= 18:  # Day
                base_temp += random.uniform(0, 3)
            else:  # Night
                base_temp -= random.uniform(0, 2)

            return round(base_temp + random.uniform(-2, 2), 1)

        elif sensor_type == 'humidity':
            base_humidity = 45.0
            if room == 'kitchen':
                base_humidity += 10.0  # Higher humidity in kitchen
            return round(base_humidity + random.uniform(-15, 15), 1)

        elif sensor_type == 'motion':
            # Motion depends on time and room
            if room == 'bedroom' and (22 <= hour or hour <= 6):
                return random.choice([0, 0, 0, 1])  # Less motion at night
            elif room == 'kitchen' and 7 <= hour <= 9:
                return random.choice([1, 1, 0])  # More motion during breakfast
            else:
                return random.choice([0, 0, 1])  # General activity

        elif sensor_type == 'light':
            if 6 <= hour <= 18:  # Daytime
                return round(random.uniform(300, 1000), 0)  # Natural light
            else:  # Nighttime
                if room == 'bedroom':
                    return round(random.uniform(0, 50), 0)  # Dark at night
                else:
                    return round(random.uniform(50, 300), 0)  # Artificial light

        elif sensor_type == 'air_quality':
            base_quality = 25.0  # Good air quality (lower is better)
            if room == 'kitchen':
                base_quality += random.uniform(0, 20)  # Cooking affects air quality
            return round(base_quality + random.uniform(-5, 15), 1)

        return 0.0

    def _get_sensor_unit(self, sensor_type: str) -> str:
        """Get measurement unit for sensor type"""
        units = {
            'temperature': '°C',
            'humidity': '%',
            'motion': 'binary',
            'light': 'lux',
            'air_quality': 'ppm'
        }
        return units.get(sensor_type, 'unknown')

    def get_system_status(self) -> Dict:
        """Get current system status and statistics"""
        blockchain_stats = self.blockchain_manager.get_blockchain_stats()

        return {
            'system_running': self.is_running,
            'total_data_points': len(self.processed_data),
            'total_alerts': len(self.alerts),
            'recent_alerts': len([a for a in self.alerts[-10:]]),
            'blockchain_records': blockchain_stats.get('total_records', 0),
            'anomalous_records': blockchain_stats.get('anomalous_records', 0),
            'privacy_epsilon': blockchain_stats.get('epsilon', 1.0),
            'k_anonymity': blockchain_stats.get('k_anonymity', 3),
            'encryption_method': blockchain_stats.get('encryption', 'AES-256-CBC'),
            'ml_model_trained': self.blockchain_manager.ml_trained
        }

    def get_recent_data(self, limit: int = 100) -> List[Dict]:
        """Get recent sensor data"""
        return self.processed_data[-limit:] if self.processed_data else []

    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent security alerts"""
        return self.alerts[-limit:] if self.alerts else []

    def get_privacy_analysis(self) -> Dict:
        """Analyze privacy preservation effectiveness"""
        if not self.processed_data:
            return {}

        df = pd.DataFrame(self.processed_data)

        # Privacy metrics
        unique_sensors = df['sensor_id'].nunique()
        unique_rooms = df['room'].nunique()
        data_points_per_sensor = len(df) / unique_sensors if unique_sensors > 0 else 0

        # Temporal distribution analysis
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        temporal_distribution = df.groupby('hour').size().to_dict()

        return {
            'total_sensors': unique_sensors,
            'total_rooms': unique_rooms,
            'avg_data_per_sensor': round(data_points_per_sensor, 2),
            'temporal_distribution': temporal_distribution,
            'privacy_techniques': {
                'differential_privacy': f"ε = {self.config['privacy_epsilon']}",
                'k_anonymity': f"k = {self.config['k_anonymity_level']}",
                'encryption': 'AES-256-CBC'
            }
        }

def create_streamlit_dashboard():
    """Create Streamlit dashboard for the IoT system"""

    st.set_page_config(
        page_title="IoT Privacy & Security Dashboard",
        page_icon="🔒",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🔒 IoT-Based Privacy-Preserving Data Management")
    st.markdown("### With Blockchain Integration and ML Intrusion Detection")

    # Initialize system if not in session state
    if 'iot_system' not in st.session_state:
        st.session_state.iot_system = IoTPrivacySecuritySystem()

    system = st.session_state.iot_system

    # Sidebar controls
    st.sidebar.title("System Control")

    if st.sidebar.button("🚀 Start System", disabled=system.is_running):
        system.start_system()
        st.sidebar.success("System started!")
        st.rerun()

    if st.sidebar.button("🛑 Stop System", disabled=not system.is_running):
        system.stop_system()
        st.sidebar.info("System stopped")
        st.rerun()

    # System status
    status = system.get_system_status()

    st.sidebar.subheader("📊 System Status")
    st.sidebar.metric("System", "🟢 Running" if status['system_running'] else "🔴 Stopped")
    st.sidebar.metric("Data Points", status['total_data_points'])
    st.sidebar.metric("Security Alerts", status['total_alerts'])
    st.sidebar.metric("Blockchain Records", status['blockchain_records'])

    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Real-Time Monitoring",
        "⛓️ Blockchain Analytics",
        "🔒 Privacy Analysis",
        "🚨 Security Alerts",
        "🤖 ML Intrusion Detection"
    ])

    with tab1:
        st.header("📊 Real-Time IoT Sensor Monitoring")

        # Recent data
        recent_data = system.get_recent_data(50)

        if recent_data:
            df = pd.DataFrame(recent_data)

            # Time series chart
            st.subheader("Sensor Data Over Time")

            # Group by sensor type for visualization
            sensor_types = df['sensor_type'].unique()

            for sensor_type in sensor_types:
                sensor_df = df[df['sensor_type'] == sensor_type]

                if not sensor_df.empty:
                    fig = px.line(
                        sensor_df,
                        x='timestamp',
                        y='value',
                        color='room',
                        title=f"{sensor_type.title()} Readings by Room",
                        labels={'value': f'{sensor_type} ({sensor_df["unit"].iloc[0]})'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Current values table
            st.subheader("Latest Sensor Readings")
            latest_df = df.groupby(['room', 'sensor_type']).last().reset_index()
            st.dataframe(latest_df[['room', 'sensor_type', 'value', 'unit', 'timestamp']],
                        use_container_width=True)
        else:
            st.info("No sensor data available. Start the system to begin data collection.")

    with tab2:
        st.header("⛓️ Blockchain Analytics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", status['blockchain_records'])
        with col2:
            st.metric("Anomalous Records", status['anomalous_records'])
        with col3:
            anomaly_rate = (status['anomalous_records'] / status['blockchain_records'] * 100) if status['blockchain_records'] > 0 else 0
            st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
        with col4:
            st.metric("ML Model", "✅ Trained" if status['ml_model_trained'] else "❌ Training")

        # Blockchain transaction visualization
        recent_data = system.get_recent_data(100)
        if recent_data:
            st.subheader("Recent Blockchain Transactions")

            df = pd.DataFrame(recent_data)
            df['block_time'] = pd.to_datetime(df['timestamp'])

            # Transaction frequency over time
            fig = px.histogram(
                df,
                x='block_time',
                nbins=20,
                title="Blockchain Transaction Frequency",
                labels={'count': 'Transactions', 'block_time': 'Time'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Transaction hash display
            st.subheader("Recent Transaction Hashes")
            tx_df = df[['timestamp', 'sensor_id', 'room', 'tx_hash']].tail(10)
            st.dataframe(tx_df, use_container_width=True)

    with tab3:
        st.header("🔒 Privacy Analysis")

        privacy_analysis = system.get_privacy_analysis()

        if privacy_analysis:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Privacy Techniques")
                st.write(f"**Differential Privacy:** {privacy_analysis['privacy_techniques']['differential_privacy']}")
                st.write(f"**K-Anonymity:** {privacy_analysis['privacy_techniques']['k_anonymity']}")
                st.write(f"**Encryption:** {privacy_analysis['privacy_techniques']['encryption']}")

                st.subheader("Data Distribution")
                st.metric("Total Sensors", privacy_analysis['total_sensors'])
                st.metric("Total Rooms", privacy_analysis['total_rooms'])
                st.metric("Avg Data/Sensor", privacy_analysis['avg_data_per_sensor'])

            with col2:
                st.subheader("Temporal Privacy Distribution")
                if privacy_analysis['temporal_distribution']:
                    temporal_df = pd.DataFrame(
                        list(privacy_analysis['temporal_distribution'].items()),
                        columns=['Hour', 'Data Points']
                    )

                    fig = px.bar(
                        temporal_df,
                        x='Hour',
                        y='Data Points',
                        title="Data Collection by Hour (Privacy Pattern Analysis)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for privacy analysis.")

    with tab4:
        st.header("🚨 Security Alerts & Intrusion Detection")

        recent_alerts = system.get_recent_alerts(20)

        if recent_alerts:
            # Alert summary
            high_alerts = len([a for a in recent_alerts if a['severity'] == 'HIGH'])
            medium_alerts = len([a for a in recent_alerts if a['severity'] == 'MEDIUM'])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔴 High Severity", high_alerts)
            with col2:
                st.metric("🟡 Medium Severity", medium_alerts)
            with col3:
                st.metric("📊 Total Alerts", len(recent_alerts))

            # Alerts timeline
            st.subheader("Security Alerts Timeline")
            alerts_df = pd.DataFrame(recent_alerts)

            fig = px.scatter(
                alerts_df,
                x='timestamp',
                y='confidence',
                color='severity',
                size='confidence',
                hover_data=['sensor_id', 'room', 'value'],
                title="Intrusion Detection Alerts Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Alerts table
            st.subheader("Recent Security Alerts")
            display_df = alerts_df[['timestamp', 'sensor_id', 'room', 'sensor_type', 'value', 'confidence', 'severity']]
            st.dataframe(display_df, use_container_width=True)
        else:
            st.success("✅ No security alerts detected")

    with tab5:
        st.header("🤖 Machine Learning Intrusion Detection")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("ML Model Status")
            if status['ml_model_trained']:
                st.success("✅ Isolation Forest model is trained and active")
                st.write("**Algorithm:** Isolation Forest")
                st.write("**Training Data:** IoT sensor readings")
                st.write("**Detection Method:** Unsupervised anomaly detection")
                st.write("**Contamination Rate:** 10% expected anomalies")
            else:
                st.warning("⚠️ ML model is still training")
                st.write("Need minimum 50 data points for training")

        with col2:
            st.subheader("Detection Performance")
            if recent_alerts:
                # Calculate detection metrics
                total_data = len(system.get_recent_data())
                total_alerts = len(recent_alerts)
                detection_rate = (total_alerts / total_data * 100) if total_data > 0 else 0

                st.metric("Detection Rate", f"{detection_rate:.2f}%")
                st.metric("Recent Anomalies", len(recent_alerts))

                # Confidence distribution
                if recent_alerts:
                    confidence_values = [alert['confidence'] for alert in recent_alerts]
                    avg_confidence = np.mean(confidence_values)
                    st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            else:
                st.info("No anomalies detected yet")

        # Feature importance and model insights
        st.subheader("ML Model Insights")
        st.write("""
        **Features Used for Detection:**
        - Hour of day (temporal patterns)
        - Day of week (weekly patterns)
        - Sensor value (statistical outliers)
        - Sensor type (context-aware detection)

        **Anomaly Indicators:**
        - Unusual sensor values for time of day
        - Irregular activity patterns
        - Statistical outliers in readings
        - Context-inappropriate measurements
        """)

    # Auto-refresh
    if status['system_running']:
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    # Enhanced Streamlit application
    try:
        main()
    except ImportError:
        # Run in console mode
        logger.info("Starting IoT Privacy & Security System in console mode...")

        system = IoTPrivacySecuritySystem()
        system.start_system()

        try:
            while True:
                time.sleep(10)
                status = system.get_system_status()
                logger.info(f"Status: {status['total_data_points']} data points, "
                          f"{status['total_alerts']} alerts")
        except KeyboardInterrupt:
            logger.info("Stopping system...")
            system.stop_system()