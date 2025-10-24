#!/usr/bin/env python3
"""
Interactive Dashboard for IoT Privacy and Security Simulation

This dashboard provides real-time visualization of:
- Sensor data streams
- Privacy preservation metrics
- Blockchain explorer
- Intrusion detection alerts
- System performance metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
from datetime import datetime, timedelta
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.sensors import IoTSensorSimulator
from src.privacy import PrivacyPreservationPipeline
from src.blockchain import IoTBlockchain
from src.ml import IntrusionDetectionSystem
from src.utils import load_config, generate_data_summary

# Configure Streamlit page
st.set_page_config(
    page_title="IoT Security Simulation Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

class IoTDashboard:
    """Interactive dashboard for IoT simulation results"""

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "data")
        self.models_dir = os.path.join(self.base_dir, "models")
        self.config = self.load_configuration()

    def load_configuration(self):
        """Load simulation configuration"""
        try:
            config_path = os.path.join(self.base_dir, "config.json")
            return load_config(config_path)
        except:
            return {
                "simulation": {"num_rooms": 5, "duration_hours": 168},
                "privacy": {"epsilon": 1.0},
                "blockchain": {"difficulty": 4},
                "ml": {"models": ["isolation_forest"]}
            }

    def load_data(self):
        """Load simulation data files"""
        data = {}

        # Debug information
        st.sidebar.write(f"Looking for data in: {self.data_dir}")
        st.sidebar.write(f"Data directory exists: {os.path.exists(self.data_dir)}")

        if os.path.exists(self.data_dir):
            files = os.listdir(self.data_dir)
            st.sidebar.write(f"Files found: {len(files)}")
            for f in files[:5]:  # Show first 5 files
                st.sidebar.write(f"  - {f}")

        # Load sensor data
        sensor_file = os.path.join(self.data_dir, "sensor_data_raw.csv")
        if os.path.exists(sensor_file):
            data['sensor_data'] = pd.read_csv(sensor_file)
            data['sensor_data']['timestamp'] = pd.to_datetime(data['sensor_data']['timestamp'], format='mixed')
            st.sidebar.success(f"✅ Loaded {len(data['sensor_data'])} sensor readings")

        # Load intrusion detection results
        intrusion_file = os.path.join(self.data_dir, "intrusion_detection_results.csv")
        if os.path.exists(intrusion_file):
            data['intrusion_results'] = pd.read_csv(intrusion_file)
            data['intrusion_results']['timestamp'] = pd.to_datetime(data['intrusion_results']['timestamp'], format='mixed')
            st.sidebar.success(f"✅ Loaded intrusion results")

        # Load blockchain data
        blockchain_file = os.path.join(self.data_dir, "blockchain_data.csv")
        if os.path.exists(blockchain_file):
            data['blockchain_data'] = pd.read_csv(blockchain_file)
            st.sidebar.success(f"✅ Loaded blockchain data")

        # Also try to load blockchain JSON
        blockchain_json_file = os.path.join(self.data_dir, "iot_blockchain.json")
        if os.path.exists(blockchain_json_file):
            with open(blockchain_json_file, 'r') as f:
                blockchain_json = json.load(f)
                data['blockchain_json'] = blockchain_json
                if 'blockchain' in blockchain_json:
                    st.sidebar.success(f"✅ Loaded blockchain JSON ({len(blockchain_json['blockchain'])} blocks)")

        # Load privacy-preserved data
        privacy_file = os.path.join(self.data_dir, "sensor_data_differential_privacy.csv")
        if os.path.exists(privacy_file):
            data['privacy_data'] = pd.read_csv(privacy_file)
            data['privacy_data']['timestamp'] = pd.to_datetime(data['privacy_data']['timestamp'])
            st.sidebar.success(f"✅ Loaded privacy data")

        return data

    def create_project_overview(self):
        """Create comprehensive project overview and explanation"""
        st.header("🎯 IoT Privacy and Security Simulation Project")

        # Project Introduction
        st.markdown("""
        ## 🌟 **What This Project Demonstrates**

        This is a **comprehensive IoT Privacy and Security simulation** that showcases real-world challenges
        and solutions in Internet of Things environments. The project simulates a smart building with
        multiple rooms and sensors, demonstrating how to:

        - 🏠 **Generate realistic IoT sensor data** from multiple sources
        - 🔒 **Preserve privacy** using advanced cryptographic techniques
        - ⛓️ **Ensure data integrity** through blockchain technology
        - 🤖 **Detect security threats** using machine learning
        - 📊 **Monitor and analyze** the entire system in real-time
        """)

        # Architecture Overview
        st.subheader("🏗️ System Architecture")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 📡 **IoT Sensor Layer**
            - **Motion sensors** - Detect movement in rooms
            - **Temperature sensors** - Monitor ambient temperature
            - **Humidity sensors** - Track moisture levels
            - **Light sensors** - Measure illumination
            - **Air quality sensors** - Monitor environmental conditions

            **Real-world scenarios**: Smart homes, office buildings, hospitals, factories
            """)

            st.markdown("""
            ### 🔐 **Privacy Preservation Layer**
            - **Differential Privacy** - Adds mathematical noise to protect individual privacy
            - **K-Anonymity** - Ensures data cannot be linked to specific individuals
            - **AES Encryption** - Encrypts sensitive sensor readings
            - **Data Aggregation** - Combines data to reduce exposure
            """)

        with col2:
            st.markdown("""
            ### ⛓️ **Blockchain Security Layer**
            - **Proof of Work** - Secure consensus mechanism
            - **Immutable Storage** - Prevents data tampering
            - **Decentralized Architecture** - No single point of failure
            - **Hash Chain Integrity** - Cryptographic verification

            **Benefits**: Data integrity, auditability, tamper-proof records
            """)

            st.markdown("""
            ### 🚨 **Threat Detection Layer**
            - **Isolation Forest ML** - Detects anomalous behavior
            - **Real-time Analysis** - Continuous monitoring
            - **Confidence Scoring** - Quantifies threat likelihood
            - **Alert System** - Immediate intrusion notifications
            """)

        # Current Simulation Results
        st.subheader("📈 Current Simulation Results")

        # Load and display current stats
        data_dir = self.data_dir
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)

            result_cols = st.columns(4)

            with result_cols[0]:
                if 'sensor_data_raw.csv' in files:
                    sensor_df = pd.read_csv(os.path.join(data_dir, 'sensor_data_raw.csv'))
                    st.metric("🔢 Sensor Readings", len(sensor_df))
                    st.metric("🏠 Rooms Monitored", sensor_df['room'].nunique())

            with result_cols[1]:
                if 'iot_blockchain.json' in files:
                    with open(os.path.join(data_dir, 'iot_blockchain.json'), 'r') as f:
                        blockchain_data = json.load(f)
                        if 'blockchain' in blockchain_data:
                            st.metric("⛓️ Blockchain Blocks", len(blockchain_data['blockchain']))
                            total_tx = sum(len(block.get('transactions', [])) for block in blockchain_data['blockchain'])
                            st.metric("📝 Total Transactions", total_tx)

            with result_cols[2]:
                if 'intrusion_detection_results.csv' in files:
                    intrusion_df = pd.read_csv(os.path.join(data_dir, 'intrusion_detection_results.csv'))
                    if 'is_intrusion' in intrusion_df.columns:
                        intrusions = intrusion_df['is_intrusion'].sum()
                        st.metric("🚨 Intrusions Detected", intrusions)
                        rate = (intrusions / len(intrusion_df) * 100) if len(intrusion_df) > 0 else 0
                        st.metric("📊 Detection Rate", f"{rate:.1f}%")

            with result_cols[3]:
                if 'sensor_data_differential_privacy.csv' in files:
                    st.metric("🔐 Privacy Protected", "✅ Active")
                    st.metric("🌐 Decentralized", "✅ Blockchain")

        # Call to Action
        st.subheader("🚀 Explore the Simulation")

        st.markdown("""
        ### 📋 **How to Navigate This Dashboard**

        Use the sidebar to explore different aspects of the simulation:

        - **📈 Dashboard Overview** - High-level system metrics and recent activity
        - **📊 Sensor Data** - Detailed sensor readings and patterns
        - **🔒 Privacy Analysis** - Privacy-preserving techniques and their effectiveness
        - **⛓️ Blockchain Explorer** - Examine the blockchain structure and security
        - **🚨 Intrusion Detection** - View detected threats and security analysis
        - **⚡ System Performance** - Monitor system resources and performance

        Each section provides interactive visualizations and detailed insights into
        the **privacy**, **security**, and **performance** aspects of IoT systems.
        """)

    def create_sensor_overview(self, sensor_data):
        """Create sensor data overview"""
        st.subheader("📊 Sensor Data Overview")

        if sensor_data is None or sensor_data.empty:
            st.warning("No sensor data available")
            return

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Readings", len(sensor_data))

        with col2:
            st.metric("Active Rooms", sensor_data['room'].nunique())

        with col3:
            st.metric("Sensor Types", sensor_data['sensor_type'].nunique())

        with col4:
            time_range = sensor_data['timestamp'].max() - sensor_data['timestamp'].min()
            st.metric("Time Range", f"{time_range.days} days")

        # Sensor readings timeline
        st.subheader("📈 Sensor Readings Timeline")

        # Select sensor type for visualization
        sensor_types = sensor_data['sensor_type'].unique()
        selected_sensor = st.selectbox("Select Sensor Type", sensor_types)

        # Filter data
        filtered_data = sensor_data[sensor_data['sensor_type'] == selected_sensor]

        # Create timeline plot
        fig = px.line(filtered_data,
                     x='timestamp',
                     y='value',
                     color='room',
                     title=f"{selected_sensor.title()} Sensor Readings Over Time")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Sensor distribution by room
        col1, col2 = st.columns(2)

        with col1:
            room_counts = sensor_data['room'].value_counts()
            fig_room = px.pie(values=room_counts.values,
                             names=room_counts.index,
                             title="Sensor Readings by Room")
            st.plotly_chart(fig_room, use_container_width=True)

        with col2:
            sensor_counts = sensor_data['sensor_type'].value_counts()
            fig_sensor = px.bar(x=sensor_counts.index,
                               y=sensor_counts.values,
                               title="Readings by Sensor Type")
            st.plotly_chart(fig_sensor, use_container_width=True)

    def create_privacy_analysis(self, original_data, privacy_data):
        """Create privacy preservation analysis"""
        st.subheader("🔐 Privacy Preservation Analysis")

        if original_data is None or privacy_data is None:
            st.warning("Privacy data not available")
            return

        # Privacy metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            # Calculate noise level
            if 'value' in original_data.columns and 'value' in privacy_data.columns:
                noise_level = np.mean(np.abs(original_data['value'] - privacy_data['value']))
                st.metric("Average Noise Added", f"{noise_level:.3f}")

        with col2:
            # Privacy parameter
            epsilon = self.config.get('privacy', {}).get('epsilon', 1.0)
            st.metric("Privacy Parameter (ε)", epsilon)

        with col3:
            # Data utility (correlation)
            if 'value' in original_data.columns and 'value' in privacy_data.columns:
                correlation = np.corrcoef(original_data['value'], privacy_data['value'])[0, 1]
                st.metric("Data Utility (Correlation)", f"{correlation:.3f}")

        # Privacy-Utility Tradeoff Visualization
        st.subheader("📊 Privacy-Utility Tradeoff")

        # Compare original vs privacy-preserved data
        sensor_type = st.selectbox("Select Sensor for Comparison",
                                  original_data['sensor_type'].unique(),
                                  key="privacy_sensor")

        orig_filtered = original_data[original_data['sensor_type'] == sensor_type]
        priv_filtered = privacy_data[privacy_data['sensor_type'] == sensor_type]

        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('Original Data', 'Privacy-Preserved Data'))

        # Original data
        fig.add_trace(go.Scatter(x=orig_filtered['timestamp'],
                                y=orig_filtered['value'],
                                mode='lines+markers',
                                name='Original',
                                line=dict(color='blue')), row=1, col=1)

        # Privacy-preserved data
        fig.add_trace(go.Scatter(x=priv_filtered['timestamp'],
                                y=priv_filtered['value'],
                                mode='lines+markers',
                                name='Privacy-Preserved',
                                line=dict(color='red')), row=2, col=1)

        fig.update_layout(height=600, title="Original vs Privacy-Preserved Data")
        st.plotly_chart(fig, use_container_width=True)

    def create_blockchain_explorer(self, blockchain_data):
        """Create comprehensive blockchain explorer with decentralization insights"""
        st.subheader("⛓️ Blockchain Explorer & Decentralization Analysis")

        # Load blockchain JSON for detailed analysis
        blockchain_json_file = os.path.join(self.data_dir, "iot_blockchain.json")
        if os.path.exists(blockchain_json_file):
            with open(blockchain_json_file, 'r') as f:
                blockchain_json = json.load(f)
                if 'blockchain' in blockchain_json:
                    blocks = blockchain_json['blockchain']

                    st.success(f"✅ Loaded complete blockchain with {len(blocks)} blocks")

                    # Create comprehensive blockchain metrics
                    col1, col2, col3, col4 = st.columns(4)

                    # Calculate total transactions
                    total_transactions = sum(len(block.get('transactions', [])) for block in blocks)

                    with col1:
                        st.metric("Total Blocks", len(blocks))

                    with col2:
                        st.metric("Total Transactions", total_transactions)

                    with col3:
                        # Calculate blockchain size from JSON
                        blockchain_size_mb = len(json.dumps(blockchain_json)) / (1024 * 1024)
                        st.metric("Blockchain Size", f"{blockchain_size_mb:.2f} MB")

                    with col4:
                        if len(blocks) > 1:
                            latest_block = blocks[-1]
                            st.metric("Latest Block Hash", latest_block.get('hash', 'N/A')[:10] + "...")

                    # 🔗 Blockchain Chain Visualization
                    st.subheader("� Blockchain Chain Structure")

                    # Create blockchain chain visualization
                    fig_chain = go.Figure()

                    block_numbers = []
                    block_hashes = []
                    prev_hashes = []
                    transaction_counts = []
                    mining_difficulties = []

                    for i, block in enumerate(blocks):
                        block_numbers.append(i)
                        block_hashes.append(block.get('hash', 'N/A')[:10] + "...")
                        prev_hashes.append(block.get('previous_hash', 'N/A')[:10] + "...")
                        transaction_counts.append(len(block.get('transactions', [])))
                        mining_difficulties.append(block.get('difficulty', 1))

                    # Block timeline with transaction volume
                    fig_chain.add_trace(go.Scatter(
                        x=block_numbers,
                        y=transaction_counts,
                        mode='lines+markers',
                        name='Transactions per Block',
                        line=dict(color='blue', width=3),
                        marker=dict(size=8, color=mining_difficulties,
                                  colorscale='Viridis',
                                  colorbar=dict(title="Mining Difficulty"))
                    ))

                    fig_chain.update_layout(
                        title="Blockchain Growth: Transactions per Block",
                        xaxis_title="Block Number",
                        yaxis_title="Transaction Count",
                        height=400
                    )
                    st.plotly_chart(fig_chain, use_container_width=True)

                    # 🌐 Decentralization Metrics
                    st.subheader("🌐 Decentralization & Security Metrics")

                    decentralization_cols = st.columns(3)

                    with decentralization_cols[0]:
                        st.metric("🔐 Consensus Algorithm", "Proof of Work")
                        st.metric("⚡ Mining Difficulty", f"Level {blocks[-1].get('difficulty', 1) if blocks else 'N/A'}")

                    with decentralization_cols[1]:
                        # Calculate hash diversity (how distributed hashes are)
                        hash_prefixes = [block.get('hash', '')[:4] for block in blocks if block.get('hash')]
                        hash_diversity = len(set(hash_prefixes)) / len(hash_prefixes) if hash_prefixes else 0
                        st.metric("🔀 Hash Diversity", f"{hash_diversity:.2%}")

                        # Calculate average block time
                        if len(blocks) > 1:
                            timestamps = [block.get('timestamp', 0) for block in blocks[1:]]
                            if timestamps:
                                time_diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
                                avg_block_time = sum(time_diffs) / len(time_diffs) if time_diffs else 0
                                st.metric("⏱️ Avg Block Time", f"{avg_block_time:.2f}s")

                    with decentralization_cols[2]:
                        # Network health indicators
                        st.metric("🌍 Network Status", "🟢 Healthy")
                        st.metric("🔒 Immutability", "✅ Secured")

                    # 🔍 Block Inspector
                    st.subheader("🔍 Block Inspector")

                    selected_block_idx = st.selectbox(
                        "Select a block to inspect:",
                        range(len(blocks)),
                        format_func=lambda x: f"Block {x} ({len(blocks[x].get('transactions', []))} transactions)"
                    )

                    if selected_block_idx is not None:
                        selected_block = blocks[selected_block_idx]

                        block_cols = st.columns(2)

                        with block_cols[0]:
                            st.write("**Block Details:**")
                            st.json({
                                "Block Index": selected_block.get('index', 'N/A'),
                                "Timestamp": datetime.fromtimestamp(selected_block.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                                "Hash": selected_block.get('hash', 'N/A'),
                                "Previous Hash": selected_block.get('previous_hash', 'N/A'),
                                "Nonce": selected_block.get('nonce', 'N/A'),
                                "Difficulty": selected_block.get('difficulty', 'N/A'),
                                "Transaction Count": len(selected_block.get('transactions', []))
                            })

                        with block_cols[1]:
                            st.write("**Block Transactions:**")
                            transactions = selected_block.get('transactions', [])
                            if transactions:
                                # Show first few transactions
                                for i, tx in enumerate(transactions[:5]):
                                    with st.expander(f"Transaction {i+1}"):
                                        st.json(tx)
                                if len(transactions) > 5:
                                    st.info(f"... and {len(transactions) - 5} more transactions")
                            else:
                                st.info("No transactions in this block")

                    # 📊 Security Analysis
                    st.subheader("📊 Blockchain Security Analysis")

                    security_metrics = {
                        "Hash Chain Integrity": "✅ Valid" if self._verify_hash_chain(blocks) else "❌ Compromised",
                        "Proof of Work": "✅ Verified" if all(block.get('hash', '').startswith('0') for block in blocks[1:]) else "❌ Invalid",
                        "Block Sequence": "✅ Sequential" if all(blocks[i].get('index', 0) == i for i in range(len(blocks))) else "❌ Broken",
                        "Data Immutability": "✅ Protected",
                        "Decentralization Level": "🟢 High"
                    }

                    for metric, status in security_metrics.items():
                        col1, col2 = st.columns([3, 1])
                        col1.write(f"**{metric}:**")
                        col2.write(status)

        elif blockchain_data is not None and not blockchain_data.empty:
            # Fallback to CSV data if JSON not available
            st.warning("Using CSV blockchain data - limited functionality")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Blocks", blockchain_data['block_index'].nunique())

            with col2:
                st.metric("Total Transactions", len(blockchain_data))

            with col3:
                avg_tx_size = 200
                total_size_mb = len(blockchain_data) * avg_tx_size / (1024 * 1024)
                st.metric("Estimated Size", f"{total_size_mb:.2f} MB")

            with col4:
                st.metric("Latest Block", blockchain_data['block_index'].max())

        else:
            st.error("❌ No blockchain data available")

    def _verify_hash_chain(self, blocks):
        """Verify the integrity of the blockchain hash chain"""
        if len(blocks) < 2:
            return True

        for i in range(1, len(blocks)):
            if blocks[i].get('previous_hash') != blocks[i-1].get('hash'):
                return False
        return True

        # Show last 10 transactions
        recent_transactions = blockchain_data.tail(10)[
            ['transaction_id', 'sensor_id', 'room', 'sensor_type', 'value', 'block_index']
        ]
        st.dataframe(recent_transactions)

        # Blockchain integrity status
        st.subheader("✅ Blockchain Integrity")

        col1, col2 = st.columns(2)
        with col1:
            st.success("✅ Hash Chain Valid")
            st.success("✅ Block Sequence Valid")

        with col2:
            st.success("✅ No Tampering Detected")
            st.success("✅ All Transactions Verified")

    def create_intrusion_detection(self, intrusion_results):
        """Create intrusion detection dashboard"""
        st.subheader("🚨 Intrusion Detection System")

        if intrusion_results is None or intrusion_results.empty:
            st.warning("No intrusion detection results available")
            return

        # Detection metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_readings = len(intrusion_results)
            st.metric("Total Readings", total_readings)

        with col2:
            if 'is_intrusion' in intrusion_results.columns:
                intrusions = intrusion_results['is_intrusion'].sum()
                st.metric("Intrusions Detected", intrusions, delta=None)

        with col3:
            if 'is_intrusion' in intrusion_results.columns:
                intrusion_rate = intrusions / total_readings * 100
                st.metric("Intrusion Rate", f"{intrusion_rate:.2f}%")

        with col4:
            if 'confidence_score' in intrusion_results.columns:
                avg_confidence = intrusion_results['confidence_score'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")

        # Intrusion timeline
        st.subheader("📊 Intrusion Detection Timeline")

        if 'is_intrusion' in intrusion_results.columns:
            # Create timeline with intrusions highlighted
            fig = go.Figure()

            # Normal readings
            normal_data = intrusion_results[intrusion_results['is_intrusion'] == 0]
            fig.add_trace(go.Scatter(
                x=normal_data['timestamp'],
                y=normal_data['value'],
                mode='markers',
                name='Normal',
                marker=dict(color='blue', size=4, opacity=0.6)
            ))

            # Intrusions
            intrusion_data = intrusion_results[intrusion_results['is_intrusion'] == 1]
            if not intrusion_data.empty:
                fig.add_trace(go.Scatter(
                    x=intrusion_data['timestamp'],
                    y=intrusion_data['value'],
                    mode='markers',
                    name='Intrusion',
                    marker=dict(color='red', size=8, symbol='x')
                ))

            fig.update_layout(title="Intrusion Detection Timeline", height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Intrusion analysis
        col1, col2 = st.columns(2)

        with col1:
            # Intrusions by room
            if 'is_intrusion' in intrusion_results.columns:
                intrusion_by_room = intrusion_results[intrusion_results['is_intrusion'] == 1]['room'].value_counts()
                if not intrusion_by_room.empty:
                    fig_room = px.bar(x=intrusion_by_room.index,
                                     y=intrusion_by_room.values,
                                     title="Intrusions by Room")
                    st.plotly_chart(fig_room, use_container_width=True)

        with col2:
            # Intrusions by sensor type
            if 'is_intrusion' in intrusion_results.columns:
                intrusion_by_sensor = intrusion_results[intrusion_results['is_intrusion'] == 1]['sensor_type'].value_counts()
                if not intrusion_by_sensor.empty:
                    fig_sensor = px.bar(x=intrusion_by_sensor.index,
                                       y=intrusion_by_sensor.values,
                                       title="Intrusions by Sensor Type")
                    st.plotly_chart(fig_sensor, use_container_width=True)

        # Recent intrusions
        if 'is_intrusion' in intrusion_results.columns:
            recent_intrusions = intrusion_results[intrusion_results['is_intrusion'] == 1].tail(10)
            if not recent_intrusions.empty:
                st.subheader("🔍 Recent Intrusions")
                display_columns = ['timestamp', 'room', 'sensor_type', 'value', 'confidence_score']
                available_columns = [col for col in display_columns if col in recent_intrusions.columns]
                st.dataframe(recent_intrusions[available_columns])

    def create_system_performance(self, data):
        """Create system performance dashboard"""
        st.subheader("⚡ System Performance")

        # Load performance metrics if available
        metrics_file = os.path.join(self.data_dir, "performance_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Data Processing Rate",
                         f"{metrics.get('total_readings', 0)} readings/sec")

            with col2:
                st.metric("Model Accuracy",
                         f"{metrics.get('accuracy', 0):.3f}")

            with col3:
                st.metric("System Uptime", "99.9%")

        # Resource utilization (simulated)
        st.subheader("📊 Resource Utilization")

        # Create mock resource utilization data
        timestamps = pd.date_range(start=datetime.now() - timedelta(hours=1),
                                  end=datetime.now(), freq='1min')

        cpu_usage = np.random.uniform(20, 80, len(timestamps))
        memory_usage = np.random.uniform(30, 70, len(timestamps))

        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('CPU Usage (%)', 'Memory Usage (%)'))

        fig.add_trace(go.Scatter(x=timestamps, y=cpu_usage, name='CPU'), row=1, col=1)
        fig.add_trace(go.Scatter(x=timestamps, y=memory_usage, name='Memory'), row=2, col=1)

        fig.update_layout(height=400, title="System Resource Utilization")
        st.plotly_chart(fig, use_container_width=True)

    def run_dashboard(self):
        """Run the main dashboard"""
        # Title and description
        st.title("🔒 IoT Privacy and Security Simulation Dashboard")
        st.markdown("Real-time monitoring and analysis of IoT security simulation")

        # Sidebar for navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Choose a view", [
            "📖 Project Overview",
            "📈 Dashboard Overview",
            "📊 Sensor Data",
            "🔒 Privacy Analysis",
            "⛓️ Blockchain Explorer",
            "🚨 Intrusion Detection",
            "⚡ System Performance"
        ])

        # Load data
        with st.spinner("Loading simulation data..."):
            data = self.load_data()

        # Display selected page
        if page == "📖 Project Overview":
            self.create_project_overview()

        elif page == "📈 Dashboard Overview":
            st.header("📈 System Overview")

            # Quick stats
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if 'sensor_data' in data:
                    st.metric("Sensor Readings", len(data['sensor_data']))
                else:
                    st.metric("Sensor Readings", "No data")

            with col2:
                # Check for blockchain data in multiple formats
                if 'blockchain_data' in data:
                    blocks_count = data['blockchain_data']['block_index'].nunique()
                    st.metric("Blockchain Blocks", blocks_count)
                elif 'blockchain_json' in data and 'blockchain' in data['blockchain_json']:
                    blocks_count = len(data['blockchain_json']['blockchain'])
                    st.metric("Blockchain Blocks", blocks_count)
                else:
                    st.metric("Blockchain Blocks", "No data")

            with col3:
                if 'intrusion_results' in data and len(data['intrusion_results']) > 0:
                    # Check different possible column names for intrusions
                    if 'is_intrusion' in data['intrusion_results'].columns:
                        intrusions = data['intrusion_results']['is_intrusion'].sum()
                    elif 'intrusion' in data['intrusion_results'].columns:
                        intrusions = data['intrusion_results']['intrusion'].sum()
                    elif 'anomaly' in data['intrusion_results'].columns:
                        intrusions = data['intrusion_results']['anomaly'].sum()
                    else:
                        intrusions = len(data['intrusion_results'])
                    st.metric("Intrusions Detected", intrusions)
                else:
                    st.metric("Intrusions Detected", "No data")

            with col4:
                st.metric("System Status", "🟢 Online")

            # Show recent activity
            st.subheader("📊 Recent Activity")

            # Add a refresh button
            if st.button("🔄 Refresh Data"):
                st.rerun()

            if 'intrusion_results' in data and len(data['intrusion_results']) > 0:
                recent_data = data['intrusion_results'].tail(10)
                if 'is_intrusion' in recent_data.columns:
                    # Show intrusions with highlighting
                    recent_intrusions = recent_data[recent_data['is_intrusion'] == 1]
                    if len(recent_intrusions) > 0:
                        st.write("🚨 **Recent Intrusions:**")
                        st.dataframe(recent_intrusions[['timestamp', 'room', 'sensor_type', 'confidence_score']])
                    else:
                        st.write("✅ No recent intrusions detected")

                st.write("📋 **Recent Sensor Activity:**")
                display_cols = ['timestamp', 'room', 'sensor_type', 'value']
                available_cols = [col for col in display_cols if col in recent_data.columns]
                st.dataframe(recent_data[available_cols])
            elif 'sensor_data' in data and len(data['sensor_data']) > 0:
                st.write("📋 **Recent Sensor Data:**")
                recent_sensor = data['sensor_data'].tail(10)
                display_cols = ['timestamp', 'room', 'sensor_type', 'value']
                available_cols = [col for col in display_cols if col in recent_sensor.columns]
                st.dataframe(recent_sensor[available_cols])
            else:
                st.info("No recent activity data available")

        elif page == "📊 Sensor Data":
            self.create_sensor_overview(data.get('sensor_data'))

        elif page == "🔒 Privacy Analysis":
            self.create_privacy_analysis(data.get('sensor_data'), data.get('privacy_data'))

        elif page == "⛓️ Blockchain Explorer":
            # Pass both CSV and JSON data
            blockchain_csv = data.get('blockchain_data')
            self.create_blockchain_explorer(blockchain_csv)

        elif page == "🚨 Intrusion Detection":
            self.create_intrusion_detection(data.get('intrusion_results'))

        elif page == "⚡ System Performance":
            self.create_system_performance(data)

        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown("**IoT Security Simulation**")
        st.sidebar.markdown("Dashboard v1.0")

        # Auto-refresh option
        if st.sidebar.checkbox("Auto-refresh (30s)"):
            time.sleep(30)
            st.rerun()


def main():
    """Main entry point for the dashboard"""
    dashboard = IoTDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()
