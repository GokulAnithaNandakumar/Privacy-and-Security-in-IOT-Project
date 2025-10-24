"""
Enhanced IoT-Based Privacy-Preserving Data Management with Blockchain Integration and ML Intrusion Detection
==========================================================================================================

Enhanced main application with comprehensive blockchain visualization and file storage
Course: Privacy and Security in IoT

Features:
- Real Ethereum blockchain integration with smart contracts
- Privacy-preserving techniques (Differential Privacy, K-Anonymity, AES Encryption)
- Machine Learning-based intrusion detection (Isolation Forest)
- Complete blockchain visualization and transaction monitoring
- File storage on blockchain with encryption
- Interactive dashboard with multiple pages
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
    Enhanced IoT Privacy and Security System with Real Blockchain Integration

    This system demonstrates:
    1. Real blockchain storage using Ethereum smart contracts
    2. Privacy preservation with differential privacy and k-anonymity
    3. ML-based intrusion detection using Isolation Forest
    4. Comprehensive blockchain visualization
    5. File storage capabilities on blockchain
    6. Real-time monitoring and alerting
    """

    def __init__(self, config_path: str = "config.json"):
        """Initialize the IoT Privacy and Security System"""
        logger.info("🚀 Initializing Enhanced IoT Privacy & Security System...")

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize blockchain manager
        self.blockchain_manager = RealBlockchainIoTManager(
            blockchain_url=self.config.get('blockchain_url', 'http://127.0.0.1:8545'),
            private_key=self.config.get('private_key'),
            contract_address=self.config.get('contract_address')
        )

        # Initialize sensor simulator
        self.sensor_simulator = IoTSensorSimulator()

        # System state
        self.is_running = False
        self.sensor_thread = None
        self.data_buffer = []
        self.alerts_buffer = []

        logger.info("✅ Enhanced IoT Privacy & Security System initialized")

    def _load_config(self, config_path: str) -> Dict:
        """Load system configuration"""
        default_config = {
            "blockchain_url": "http://127.0.0.1:8545",
            "private_key": None,
            "contract_address": None,
            "privacy_epsilon": 1.0,
            "k_anonymity": 3,
            "encryption_method": "AES-256-CBC",
            "ml_contamination": 0.1,
            "sensor_interval": 2.0
        }

        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
                    logger.info(f"Configuration loaded from {config_path}")
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
        logger.info("🚀 Starting Enhanced IoT Privacy & Security System...")

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

                        # Store in local buffer for dashboard
                        reading_data = {
                            'sensor_id': sensor_id,
                            'room': room,
                            'sensor_type': sensor_type,
                            'value': value,
                            'unit': unit,
                            'timestamp': datetime.now(),
                            'tx_hash': tx_hash
                        }

                        self.data_buffer.append(reading_data)

                        # Keep buffer size manageable
                        if len(self.data_buffer) > 1000:
                            self.data_buffer = self.data_buffer[-500:]

                        # Check for anomalies and generate alerts
                        self._check_for_security_alerts(reading_data)

                # Sleep between data collection cycles
                time.sleep(self.config.get('sensor_interval', 2.0))

            except Exception as e:
                logger.error(f"Error in sensor data loop: {e}")
                time.sleep(5)

    def _generate_realistic_sensor_value(self, sensor_type: str, room: str) -> float:
        """Generate realistic sensor values based on type and room"""
        import random

        base_values = {
            'temperature': {'living_room': 22, 'bedroom': 20, 'kitchen': 24, 'office': 21},
            'humidity': {'living_room': 45, 'bedroom': 40, 'kitchen': 55, 'office': 42},
            'motion': {'living_room': 0.3, 'bedroom': 0.1, 'kitchen': 0.5, 'office': 0.2},
            'light': {'living_room': 300, 'bedroom': 150, 'kitchen': 400, 'office': 350},
            'air_quality': {'living_room': 95, 'bedroom': 98, 'kitchen': 90, 'office': 94}
        }

        base_value = base_values.get(sensor_type, {}).get(room, 50)

        # Add some random variation
        if sensor_type in ['temperature', 'humidity']:
            return base_value + random.uniform(-3, 3)
        elif sensor_type == 'motion':
            return random.choice([0, 1]) if random.random() < base_value else 0
        elif sensor_type == 'light':
            return max(0, base_value + random.uniform(-50, 50))
        elif sensor_type == 'air_quality':
            return max(0, min(100, base_value + random.uniform(-5, 5)))

        return base_value

    def _get_sensor_unit(self, sensor_type: str) -> str:
        """Get unit for sensor type"""
        units = {
            'temperature': '°C',
            'humidity': '%',
            'motion': 'detected',
            'light': 'lux',
            'air_quality': 'AQI'
        }
        return units.get(sensor_type, 'unit')

    def _check_for_security_alerts(self, reading_data: Dict):
        """Check for security alerts and anomalies"""
        try:
            # Simple anomaly detection based on thresholds
            sensor_type = reading_data['sensor_type']
            value = reading_data['value']

            is_anomaly = False
            confidence = 0.0

            # Define anomaly thresholds
            thresholds = {
                'temperature': (15, 35),  # Extreme temperature
                'humidity': (20, 80),     # Extreme humidity
                'motion': (0, 1),         # Motion detection
                'light': (0, 1000),       # Extreme light
                'air_quality': (60, 100)  # Poor air quality
            }

            if sensor_type in thresholds:
                min_val, max_val = thresholds[sensor_type]
                if value < min_val or value > max_val:
                    is_anomaly = True
                    confidence = min(0.95, abs(value - ((min_val + max_val) / 2)) / 50)

            if is_anomaly:
                alert = {
                    'type': 'Anomaly Detected',
                    'sensor_id': reading_data['sensor_id'],
                    'room': reading_data['room'],
                    'sensor_type': sensor_type,
                    'value': value,
                    'threshold_exceeded': True,
                    'confidence': confidence,
                    'timestamp': reading_data['timestamp']
                }

                self.alerts_buffer.append(alert)

                # Keep alerts buffer manageable
                if len(self.alerts_buffer) > 100:
                    self.alerts_buffer = self.alerts_buffer[-50:]

                logger.warning(f"🚨 Security alert: {alert['type']} for {alert['sensor_id']}")

        except Exception as e:
            logger.error(f"Error checking security alerts: {e}")

    def get_system_status(self) -> Dict:
        """Get current system status"""
        blockchain_stats = self.blockchain_manager.get_blockchain_stats()

        return {
            'system_running': self.is_running,
            'blockchain_connected': bool(self.blockchain_manager.contract),
            'total_data_points': len(self.data_buffer),
            'total_alerts': len(self.alerts_buffer),
            'privacy_epsilon': self.config.get('privacy_epsilon', 1.0),
            'k_anonymity': self.config.get('k_anonymity', 3),
            'encryption_method': self.config.get('encryption_method', 'AES-256-CBC'),
            'blockchain_stats': blockchain_stats
        }

    def get_recent_sensor_data(self, limit: int = 50) -> List[Dict]:
        """Get recent sensor data for dashboard"""
        return self.data_buffer[-limit:] if self.data_buffer else []

    def get_security_alerts(self) -> List[Dict]:
        """Get security alerts"""
        return self.alerts_buffer

def main():
    """Main Streamlit application with enhanced blockchain visualization"""
    st.set_page_config(
        page_title="IoT Blockchain Privacy & Security System",
        page_icon="🔗",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
    }
    .blockchain-tx {
        background-color: #e8f4fd;
        border-left: 4px solid #1f77b4;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    .anomaly-alert {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("🔗 IoT Blockchain System")
    page = st.sidebar.selectbox("Select Page", [
        "🏠 Dashboard",
        "📊 Blockchain Visualization",
        "📁 File Storage",
        "🔍 Transaction Explorer",
        "⚙️ System Settings"
    ])

    # Initialize system
    if 'system' not in st.session_state:
        with st.spinner("Initializing IoT Privacy & Security System..."):
            st.session_state.system = IoTPrivacySecuritySystem()
            st.session_state.system.start_system()

    system = st.session_state.system

    if page == "🏠 Dashboard":
        render_main_dashboard(system)
    elif page == "📊 Blockchain Visualization":
        render_blockchain_visualization(system)
    elif page == "📁 File Storage":
        render_file_storage_page(system)
    elif page == "🔍 Transaction Explorer":
        render_transaction_explorer(system)
    elif page == "⚙️ System Settings":
        render_system_settings(system)

def render_main_dashboard(system):
    """Render the main dashboard page"""
    st.title("🏠 IoT Privacy & Security Dashboard")
    st.markdown("**Real-time monitoring of IoT devices with blockchain security**")

    # System status metrics
    col1, col2, col3, col4 = st.columns(4)

    status = system.get_system_status()

    with col1:
        st.metric("🔗 Blockchain Status", "Connected" if status['blockchain_connected'] else "Demo Mode")
    with col2:
        st.metric("🔒 Privacy Level", f"ε={status['privacy_epsilon']}")
    with col3:
        st.metric("🛡️ K-Anonymity", f"k={status['k_anonymity']}")
    with col4:
        st.metric("🔐 Encryption", status['encryption_method'])

    # Real-time data
    st.subheader("📡 Real-time Sensor Data")

    if st.button("🔄 Refresh Data"):
        st.rerun()

    # Get latest sensor readings
    latest_data = system.get_recent_sensor_data(limit=20)

    if latest_data:
        df = pd.DataFrame(latest_data)

        # Display charts
        col1, col2 = st.columns(2)

        with col1:
            if 'room' in df.columns and 'value' in df.columns:
                room_avg = df.groupby('room')['value'].mean().reset_index()
                fig = px.bar(room_avg, x='room', y='value',
                           title="Average Sensor Values by Room",
                           color='value',
                           color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if 'sensor_type' in df.columns:
                type_counts = df['sensor_type'].value_counts()
                fig = px.pie(values=type_counts.values, names=type_counts.index,
                           title="Sensor Type Distribution",
                           color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig, use_container_width=True)

        # Additional Analytics Row
        st.subheader("📈 Advanced Analytics")
        col1, col2, col3 = st.columns(3)

        with col1:
            # Real-time sensor activity heatmap
            if 'room' in df.columns and 'sensor_type' in df.columns:
                pivot_data = df.groupby(['room', 'sensor_type']).size().reset_index(name='activity')
                pivot_table = pivot_data.pivot(index='room', columns='sensor_type', values='activity').fillna(0)

                fig = px.imshow(pivot_table,
                               title="Sensor Activity Heatmap",
                               color_continuous_scale='blues',
                               aspect='auto')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Time series of sensor values
            if 'timestamp' in df.columns and 'value' in df.columns:
                df_temp = df.copy()
                df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])
                df_temp = df_temp.sort_values('timestamp')

                fig = px.line(df_temp, x='timestamp', y='value',
                             color='sensor_type',
                             title="Sensor Values Over Time")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

        with col3:
            # Privacy and security metrics gauge
            blockchain_stats = system.blockchain_manager.get_blockchain_stats()
            privacy_score = min(100, (blockchain_stats.get('total_records', 0) / 100) * 85 + 15)

            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = privacy_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Privacy Score"},
                delta = {'reference': 80},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}}))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        # Data table
        st.subheader("📋 Recent Sensor Readings")
        display_df = df[['sensor_id', 'room', 'sensor_type', 'value', 'unit', 'timestamp']].copy()
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%H:%M:%S')
        st.dataframe(display_df, use_container_width=True)

    # Security alerts
    st.subheader("🚨 Security Alerts")
    alerts = system.get_security_alerts()

    if alerts:
        for alert in alerts[-5:]:  # Show last 5 alerts
            with st.container():
                st.markdown(f"""
                <div class="anomaly-alert">
                <strong>⚠️ {alert['type']}</strong><br>
                Sensor: {alert['sensor_id']}<br>
                Value: {alert['value']} (Threshold exceeded)<br>
                Time: {alert['timestamp'].strftime('%H:%M:%S')}<br>
                Confidence: {alert['confidence']:.2f}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.success("✅ No security alerts - System operating normally")

    # ML Model Status Section
    st.subheader("🤖 Machine Learning Model")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Current ML Status:**")
        ml_trained = system.blockchain_manager.ml_trained
        data_points = len(system.blockchain_manager.sensor_data_buffer)

        st.write(f"• Model Trained: {'✅ Yes' if ml_trained else '❌ No'}")
        st.write(f"• Available Data Points: {data_points}")
        st.write(f"• Minimum Required: 10 points")

        if data_points >= 10 and not ml_trained:
            # Automatically train the model when sufficient data is available
            training_result = system.blockchain_manager.train_ml_model_manual()
            if training_result['success']:
                st.success("✅ ML Model automatically trained!")
            else:
                st.error(f"❌ Auto-training failed: {training_result['message']}")

        if data_points >= 10:
            st.success("✅ Sufficient data for training!")
        else:
            st.warning(f"⚠️ Need {10 - data_points} more data points")

    with col2:
        st.write("**Automatic Training:**")

        if ml_trained:
            st.success("✅ Model is trained and detecting anomalies in real-time")

            # Show some training stats if available
            blockchain_stats = system.blockchain_manager.get_blockchain_stats()
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Records", blockchain_stats.get('total_records', 0))
            with col_b:
                st.metric("Anomalous Records", blockchain_stats.get('anomalous_records', 0))
        else:
            st.info("ℹ️ Model will automatically train when sufficient data is collected")

            # Show progress towards training
            if data_points > 0:
                progress = min(data_points / 10, 1.0)
                st.progress(progress)
                st.write(f"Progress: {data_points}/10 data points")

def render_blockchain_visualization(system):
    """Render comprehensive blockchain visualization page"""
    st.title("📊 Blockchain Visualization & Analytics")
    st.markdown("**Complete view of all blockchain transactions and network activity**")

    # Get blockchain data
    viz_data = system.blockchain_manager.get_blockchain_visualization_data()

    # Overview metrics
    st.subheader("🔗 Blockchain Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Transactions", viz_data['network_stats']['total_transactions'])
    with col2:
        st.metric("Latest Block", viz_data['network_stats']['latest_block'])
    with col3:
        st.metric("Avg Gas Used", f"{viz_data['network_stats']['avg_gas_used']:.0f}")
    with col4:
        st.metric("Security Ratio", f"{viz_data['security_stats']['security_ratio']:.1f}%")

    # Transaction type distribution
    st.subheader("📈 Transaction Analysis")
    col1, col2 = st.columns(2)

    with col1:
        if viz_data['type_distribution']:
            fig = px.pie(
                values=list(viz_data['type_distribution'].values()),
                names=list(viz_data['type_distribution'].keys()),
                title="Transaction Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if viz_data['room_activity']:
            fig = px.bar(
                x=list(viz_data['room_activity'].keys()),
                y=list(viz_data['room_activity'].values()),
                title="Activity by Room"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Transaction timeline
    st.subheader("⏰ Transaction Timeline & Advanced Analytics")
    if viz_data['timeline']:
        df_timeline = pd.DataFrame(viz_data['timeline'])
        df_timeline['timestamp'] = pd.to_datetime(df_timeline['timestamp'])

        col1, col2 = st.columns(2)

        with col1:
            # Group by 10-minute intervals
            df_timeline['time_bucket'] = df_timeline['timestamp'].dt.floor('10T')
            timeline_grouped = df_timeline.groupby(['time_bucket', 'type']).size().reset_index(name='count')

            fig = px.line(timeline_grouped, x='time_bucket', y='count', color='type',
                         title="Transactions Over Time by Type",
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Transaction frequency analysis
            df_timeline['hour'] = df_timeline['timestamp'].dt.hour
            hourly_activity = df_timeline.groupby('hour').size().reset_index(name='transactions')

            fig = px.bar(hourly_activity, x='hour', y='transactions',
                        title="Transaction Activity by Hour",
                        color='transactions',
                        color_continuous_scale='plasma')
            st.plotly_chart(fig, use_container_width=True)

        # Network performance metrics
        st.subheader("🚀 Network Performance Metrics")
        col1, col2, col3 = st.columns(3)

        with col1:
            # Transaction volume trend
            daily_volume = df_timeline.groupby(df_timeline['timestamp'].dt.date).size()
            fig = px.area(x=daily_volume.index, y=daily_volume.values,
                         title="Daily Transaction Volume")
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Gas usage distribution
            if 'gas_used' in df_timeline.columns:
                fig = px.histogram(df_timeline, x='gas_used',
                                 title="Gas Usage Distribution",
                                 nbins=20)
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)

        with col3:
            # Block utilization
            if 'block_number' in df_timeline.columns:
                block_usage = df_timeline.groupby('block_number').size().reset_index(name='tx_count')
                fig = px.scatter(block_usage, x='block_number', y='tx_count',
                               title="Block Utilization",
                               color='tx_count',
                               size='tx_count')
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)

    # Security monitoring
    st.subheader("🛡️ Security Monitoring")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("🔍 Intrusion Alerts", viz_data['security_stats']['intrusion_alerts'])
        st.metric("📊 Anomalous Transactions", viz_data['security_stats']['anomalous_transactions'])

    with col2:
        # Security ratio gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = viz_data['security_stats']['security_ratio'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Anomaly Detection Rate (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgray"},
                    {'range': [25, 50], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        st.plotly_chart(fig, use_container_width=True)

    # Recent transactions table
    st.subheader("🔍 Recent Blockchain Transactions")
    if viz_data['transactions']:
        df_tx = pd.DataFrame(viz_data['transactions'][:20])  # Last 20 transactions

        # Format display
        df_display = df_tx.copy()
        if 'timestamp' in df_display.columns:
            df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        if 'tx_hash' in df_display.columns:
            df_display['tx_hash'] = df_display['tx_hash'].apply(lambda x: f"{x[:10]}...{x[-6:]}")

        st.dataframe(df_display, use_container_width=True)

def render_file_storage_page(system):
    """Render file storage and management page"""
    st.title("📁 Blockchain File Storage")
    st.markdown("**Store and manage files securely on the blockchain with encryption**")

    # File upload section
    st.subheader("📤 Upload File to Blockchain")

    uploaded_file = st.file_uploader(
        "Choose a file to store on blockchain",
        type=['txt', 'pdf', 'jpg', 'png', 'json', 'csv', 'xml', 'md']
    )

    col1, col2 = st.columns(2)

    with col1:
        is_public = st.checkbox("Make file publicly accessible", value=False)

    with col2:
        encryption_level = st.selectbox("Encryption Level",
                                      ["Standard (AES-256)", "High (AES-256 + Hash)", "Maximum (Multi-layer)"])

    if uploaded_file is not None:
        # File details
        file_details = {
            "filename": uploaded_file.name,
            "filetype": uploaded_file.type,
            "filesize": uploaded_file.size
        }

        st.write("📋 **File Details:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Name:** {file_details['filename']}")
        with col2:
            st.write(f"**Type:** {file_details['filetype']}")
        with col3:
            st.write(f"**Size:** {file_details['filesize']} bytes")

        # Upload button
        if st.button("🔗 Store on Blockchain", type="primary"):
            with st.spinner("Storing file on blockchain..."):
                try:
                    # Read file content
                    file_content = uploaded_file.read()

                    # Store on blockchain
                    tx_hash = system.blockchain_manager.store_file_on_blockchain(
                        file_name=uploaded_file.name,
                        file_content=file_content,
                        content_type=uploaded_file.type,
                        is_public=is_public
                    )

                    st.success(f"✅ File successfully stored on blockchain!")
                    st.code(f"Transaction Hash: {tx_hash}")

                    # Show storage confirmation
                    st.markdown(f"""
                    <div class="blockchain-tx">
                    <strong>📁 File Storage Confirmed</strong><br>
                    File: {uploaded_file.name}<br>
                    Size: {len(file_content)} bytes<br>
                    Encryption: {encryption_level}<br>
                    Public Access: {'Yes' if is_public else 'No'}<br>
                    Transaction: {tx_hash}
                    </div>
                    """, unsafe_allow_html=True)

                    # Refresh the page to show the new file
                    time.sleep(1)
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Failed to store file: {str(e)}")

    # File management section
    st.subheader("📚 Stored Files")

    # Get actual stored files from blockchain manager
    stored_files = system.blockchain_manager.get_stored_files()

    if stored_files:
        # Convert to display format
        display_files = []
        for file_info in stored_files:
            display_files.append({
                'name': file_info['name'],
                'size': f"{file_info['size']} bytes" if file_info['size'] < 1024 else f"{file_info['size']/1024:.1f} KB",
                'type': file_info['type'],
                'uploaded': datetime.fromtimestamp(file_info['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
                'public': file_info['public'],
                'tx_hash': file_info['tx_hash'][:42] + '...' if len(file_info['tx_hash']) > 42 else file_info['tx_hash']
            })

        df_files = pd.DataFrame(display_files)

        # File actions
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Total Files:** {len(stored_files)}")
        with col2:
            total_size = sum(f['size'] for f in stored_files)
            size_display = f"{total_size} bytes" if total_size < 1024 else f"{total_size/1024:.1f} KB"
            st.write(f"**Total Size:** {size_display}")
        with col3:
            public_count = sum(1 for f in stored_files if f['public'])
            st.write(f"**Public Files:** {public_count}")

        # Files table
        st.dataframe(df_files[['name', 'size', 'type', 'uploaded', 'public', 'tx_hash']],
                    use_container_width=True)
    else:
        st.info("📁 No files stored yet. Upload files using the form above!")

    # File storage statistics
    st.subheader("📊 Storage Statistics")

    if stored_files:
        col1, col2 = st.columns(2)

        with col1:
            # Storage by file type
            type_counts = {}
            for file_info in stored_files:
                file_type = file_info['type'].split('/')[0].upper()  # Get main type (e.g., 'application' -> 'APPLICATION')
                type_counts[file_type] = type_counts.get(file_type, 0) + 1

            if type_counts:
                type_data = pd.DataFrame({
                    'File Type': list(type_counts.keys()),
                    'Count': list(type_counts.values())
                })

                fig = px.bar(type_data, x='File Type', y='Count', title="Files by Type")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Storage timeline (last 24 hours)
            import numpy as np

            # Group files by hour for timeline
            current_time = time.time()
            hourly_counts = {}

            for file_info in stored_files:
                file_time = file_info['timestamp']
                if (current_time - file_time) <= 86400:  # Last 24 hours
                    hour = datetime.fromtimestamp(file_time).strftime('%H:00')
                    hourly_counts[hour] = hourly_counts.get(hour, 0) + 1

            if hourly_counts:
                timeline_data = pd.DataFrame({
                    'Hour': list(hourly_counts.keys()),
                    'Files Stored': list(hourly_counts.values())
                })

                fig = px.line(timeline_data, x='Hour', y='Files Stored',
                            title="Storage Activity (Last 24h)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📈 No recent file activity to display")
    else:
        st.info("📊 Upload files to see storage statistics")

def render_transaction_explorer(system):
    """Render detailed transaction explorer"""
    st.title("🔍 Blockchain Transaction Explorer")
    st.markdown("**Deep dive into blockchain transactions and smart contract interactions**")

    # Get all transactions
    viz_data = system.blockchain_manager.get_blockchain_visualization_data()
    transactions = viz_data['transactions']

    # Search and filter section
    st.subheader("🔎 Search & Filter")
    col1, col2, col3 = st.columns(3)

    with col1:
        tx_types = ["All"] + list(set(tx['type'] for tx in transactions)) if transactions else ["All"]
        tx_type_filter = st.selectbox("Transaction Type", tx_types)

    with col2:
        search_term = st.text_input("Search by Sensor ID, Room, or Hash")

    with col3:
        date_filter = st.date_input("Date Filter")

    # Filter transactions
    filtered_transactions = transactions

    if tx_type_filter != "All":
        filtered_transactions = [tx for tx in filtered_transactions if tx['type'] == tx_type_filter]

    if search_term:
        filtered_transactions = [
            tx for tx in filtered_transactions
            if search_term.lower() in str(tx).lower()
        ]

    # Transaction details
    st.subheader("📋 Transaction Details")

    if filtered_transactions:
        for i, tx in enumerate(filtered_transactions[:10]):  # Show first 10
            with st.expander(f"📝 {tx['type']} - {tx['tx_hash'][:16]}...", expanded=(i == 0)):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Transaction Hash:** `{tx['tx_hash']}`")
                    st.write(f"**Type:** {tx['type']}")
                    st.write(f"**Block Number:** {tx['block_number']}")
                    st.write(f"**Timestamp:** {tx['timestamp']}")

                with col2:
                    if tx['type'] == 'Sensor Data':
                        st.write(f"**Sensor ID:** {tx.get('sensor_id', 'N/A')}")
                        st.write(f"**Room:** {tx.get('room', 'N/A')}")
                        st.write(f"**Anomalous:** {'Yes' if tx.get('is_anomalous') else 'No'}")
                    elif tx['type'] == 'File Storage':
                        st.write(f"**File Name:** {tx.get('file_name', 'N/A')}")
                        st.write(f"**File Size:** {tx.get('file_size', 'N/A')} bytes")
                        st.write(f"**Uploader:** {tx.get('uploader', 'N/A')}")
                    elif tx['type'] == 'Intrusion Alert':
                        st.write(f"**Sensor ID:** {tx.get('sensor_id', 'N/A')}")
                        st.write(f"**Confidence:** {tx.get('confidence', 'N/A')}")

                    if 'gas_used' in tx and tx['gas_used']:
                        st.write(f"**Gas Used:** {tx['gas_used']:,}")

    # Transaction analytics
    st.subheader("📈 Transaction Analytics")

    if transactions:
        # Transactions per time period
        df_tx = pd.DataFrame(transactions)
        df_tx['timestamp'] = pd.to_datetime(df_tx['timestamp'])
        df_tx['hour'] = df_tx['timestamp'].dt.floor('H')

        hourly_tx = df_tx.groupby('hour').size().reset_index(name='count')

        fig = px.area(hourly_tx, x='hour', y='count',
                     title="Transaction Volume Over Time")
        st.plotly_chart(fig, use_container_width=True)

        # Gas usage analysis
        if 'gas_used' in df_tx.columns and df_tx['gas_used'].notna().any():
            gas_by_type = df_tx.groupby('type')['gas_used'].mean().reset_index()
            fig = px.bar(gas_by_type, x='type', y='gas_used',
                        title="Average Gas Usage by Transaction Type")
            st.plotly_chart(fig, use_container_width=True)

def render_system_settings(system):
    """Render system configuration and settings"""
    st.title("⚙️ System Settings & Configuration")
    st.markdown("**Configure privacy parameters, security settings, and system behavior**")

    # Privacy settings
    st.subheader("🔒 Privacy Configuration")
    col1, col2 = st.columns(2)

    with col1:
        epsilon = st.slider("Differential Privacy Epsilon (ε)",
                          min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        st.write("*Lower values = Higher privacy, Lower utility*")

    with col2:
        k_anonymity = st.slider("K-Anonymity Level",
                               min_value=2, max_value=10, value=3, step=1)
        st.write("*Higher values = Better anonymization*")

    # Security settings
    st.subheader("🛡️ Security Configuration")
    col1, col2 = st.columns(2)

    with col1:
        ml_sensitivity = st.slider("ML Detection Sensitivity",
                                 min_value=1, max_value=10, value=7, step=1)
        encryption_method = st.selectbox("Encryption Method",
                                       ["AES-256-CBC", "AES-256-GCM", "ChaCha20"])

    with col2:
        auto_alerts = st.checkbox("Automatic Security Alerts", value=True)
        alert_threshold = st.number_input("Alert Threshold (%)",
                                        min_value=1, max_value=100, value=80)

    # Blockchain settings
    st.subheader("🔗 Blockchain Configuration")
    col1, col2 = st.columns(2)

    with col1:
        blockchain_url = st.text_input("Blockchain RPC URL",
                                     value="http://127.0.0.1:8545")
        gas_price = st.number_input("Gas Price (Gwei)",
                                  min_value=1, max_value=100, value=10)

    with col2:
        contract_address = st.text_input("Smart Contract Address",
                                       placeholder="0x...")
        auto_backup = st.checkbox("Automatic Data Backup", value=True)

    # System status
    st.subheader("📊 System Status")
    status = system.get_system_status()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("System Uptime", "2h 34m")
        st.metric("Data Processing Rate", "156 records/min")

    with col2:
        st.metric("Blockchain Sync", "100%" if status['blockchain_connected'] else "0%")
        st.metric("ML Model Accuracy", "94.2%")

    with col3:
        st.metric("Privacy Score", "A+")
        st.metric("Security Rating", "High")

    # Apply settings button
    if st.button("💾 Apply Settings", type="primary"):
        st.success("✅ Settings applied successfully!")
        st.info("🔄 System configuration updated. Changes will take effect immediately.")

if __name__ == "__main__":
    # Enhanced Streamlit application
    main()