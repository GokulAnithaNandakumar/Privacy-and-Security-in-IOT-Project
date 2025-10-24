#!/usr/bin/env python3
"""
Enhanced Demo Script for IoT Blockchain Privacy & Security System
================================================================

This demo showcases all the enhanced features:
1. Comprehensive blockchain visualization
2. Real-time transaction monitoring
3. File storage on blockchain
4. Security anomaly detection
5. Multi-page dashboard interface

Run this to see the complete enhanced system in action!
"""

import time
import tempfile
import os
from main_blockchain_enhanced import IoTPrivacySecuritySystem

def print_banner():
    """Print demo banner"""
    print("\n" + "="*80)
    print("🔗 ENHANCED IoT BLOCKCHAIN PRIVACY & SECURITY SYSTEM DEMO")
    print("="*80)
    print("Features: Blockchain Visualization | File Storage | Transaction Explorer")
    print("Course: Privacy and Security in IoT")
    print("="*80 + "\n")

def demo_blockchain_visualization(system):
    """Demo blockchain visualization capabilities"""
    print("📊 BLOCKCHAIN VISUALIZATION DEMO")
    print("-" * 40)

    # Get comprehensive blockchain data
    viz_data = system.blockchain_manager.get_blockchain_visualization_data()

    print(f"🔗 Total Transactions: {viz_data['network_stats']['total_transactions']}")
    print(f"📦 Latest Block: {viz_data['network_stats']['latest_block']}")
    print(f"⛽ Avg Gas Used: {viz_data['network_stats']['avg_gas_used']:.0f}")
    print(f"🛡️ Security Ratio: {viz_data['security_stats']['security_ratio']:.1f}%")

    print("\n📈 Transaction Type Distribution:")
    for tx_type, count in viz_data['type_distribution'].items():
        print(f"  • {tx_type}: {count} transactions")

    print("\n🏠 Room Activity:")
    for room, activity in viz_data['room_activity'].items():
        print(f"  • {room}: {activity} sensor readings")

    print("\n🔍 Recent Transactions:")
    for i, tx in enumerate(viz_data['transactions'][:5]):
        print(f"  {i+1}. {tx['type']} - {tx['tx_hash'][:16]}... (Block {tx['block_number']})")

    print("✅ Blockchain visualization data ready for dashboard!")

def demo_file_storage(system):
    """Demo file storage on blockchain"""
    print("\n📁 FILE STORAGE ON BLOCKCHAIN DEMO")
    print("-" * 40)

    # Create sample files to store
    files_to_store = [
        {
            "name": "iot_sensor_config.json",
            "content": '{"sensors": ["temperature", "humidity", "motion"], "rooms": ["living_room", "kitchen"]}',
            "type": "application/json",
            "public": False
        },
        {
            "name": "privacy_settings.txt",
            "content": "Differential Privacy: ε=1.0\\nK-Anonymity: k=3\\nEncryption: AES-256-CBC",
            "type": "text/plain",
            "public": True
        },
        {
            "name": "security_log.csv",
            "content": "timestamp,sensor_id,anomaly_detected,confidence\\n2024-10-24 10:30:00,kitchen_light_001,true,0.85",
            "type": "text/csv",
            "public": False
        }
    ]

    stored_files = []

    for file_info in files_to_store:
        print(f"📤 Storing {file_info['name']} on blockchain...")

        # Convert string content to bytes
        file_content = file_info['content'].encode('utf-8')

        # Store on blockchain
        tx_hash = system.blockchain_manager.store_file_on_blockchain(
            file_name=file_info['name'],
            file_content=file_content,
            content_type=file_info['type'],
            is_public=file_info['public']
        )

        stored_files.append({
            'name': file_info['name'],
            'size': len(file_content),
            'tx_hash': tx_hash,
            'public': file_info['public']
        })

        print(f"   ✅ Stored with TX: {tx_hash}")
        print(f"   📋 Size: {len(file_content)} bytes | Public: {file_info['public']}")

    print(f"\n📚 Summary: {len(stored_files)} files stored on blockchain")
    total_size = sum(f['size'] for f in stored_files)
    public_count = sum(1 for f in stored_files if f['public'])
    print(f"💾 Total Storage: {total_size} bytes")
    print(f"🌐 Public Files: {public_count}/{len(stored_files)}")

def demo_transaction_monitoring(system):
    """Demo real-time transaction monitoring"""
    print("\n🔍 REAL-TIME TRANSACTION MONITORING DEMO")
    print("-" * 40)

    print("⏰ Monitoring transactions for 10 seconds...")

    initial_data = system.get_recent_sensor_data(limit=1)
    initial_count = len(initial_data) if initial_data else 0

    # Monitor for a short period
    for i in range(5):
        print(f"📡 Monitoring cycle {i+1}/5...")
        time.sleep(2)

        # Get latest data
        current_data = system.get_recent_sensor_data(limit=5)
        alerts = system.get_security_alerts()

        if current_data:
            latest = current_data[-1]
            print(f"   📊 Latest: {latest['sensor_id']} = {latest['value']} {latest['unit']}")
            print(f"   🔗 TX: {latest['tx_hash'][:20]}...")

        if alerts:
            recent_alerts = [a for a in alerts if (time.time() - a['timestamp'].timestamp()) < 30]
            print(f"   🚨 Recent Alerts: {len(recent_alerts)}")

    # Final status
    final_data = system.get_recent_sensor_data(limit=1)
    final_count = len(system.get_recent_sensor_data(limit=1000))
    new_transactions = final_count - initial_count

    print(f"\n📈 Monitoring Results:")
    print(f"   • New Transactions: {new_transactions}")
    print(f"   • Total Sensor Readings: {final_count}")
    print(f"   • Security Alerts: {len(system.get_security_alerts())}")

def demo_security_features(system):
    """Demo security and privacy features"""
    print("\n🛡️ SECURITY & PRIVACY FEATURES DEMO")
    print("-" * 40)

    # Get system status
    status = system.get_system_status()

    print("🔒 Privacy Configuration:")
    print(f"   • Differential Privacy: ε = {status['privacy_epsilon']}")
    print(f"   • K-Anonymity Level: k = {status['k_anonymity']}")
    print(f"   • Encryption Method: {status['encryption_method']}")

    print("\n🔗 Blockchain Security:")
    print(f"   • Blockchain Connected: {status['blockchain_connected']}")
    print(f"   • Total Data Points: {status['total_data_points']}")
    if status['blockchain_stats']:
        blockchain_stats = status['blockchain_stats']
        print(f"   • Blockchain Records: {blockchain_stats.get('total_records', 0)}")
        print(f"   • Anomalous Records: {blockchain_stats.get('anomalous_records', 0)}")

    print("\n🚨 Intrusion Detection:")
    alerts = system.get_security_alerts()
    print(f"   • Total Alerts Generated: {len(alerts)}")

    if alerts:
        print("   • Recent Alert Types:")
        alert_types = {}
        for alert in alerts[-10:]:  # Last 10 alerts
            alert_type = alert['type']
            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1

        for alert_type, count in alert_types.items():
            print(f"     - {alert_type}: {count}")

        # Show latest alert
        latest_alert = alerts[-1]
        print(f"   • Latest Alert: {latest_alert['sensor_id']}")
        print(f"     Confidence: {latest_alert['confidence']:.2f}")
        print(f"     Time: {latest_alert['timestamp'].strftime('%H:%M:%S')}")

def demo_dashboard_features():
    """Demo dashboard and UI features"""
    print("\n🖥️ DASHBOARD FEATURES")
    print("-" * 40)
    print("The enhanced dashboard includes:")
    print("   🏠 Main Dashboard - Real-time monitoring")
    print("   📊 Blockchain Visualization - Complete transaction analytics")
    print("   📁 File Storage - Upload and manage files on blockchain")
    print("   🔍 Transaction Explorer - Deep dive into blockchain data")
    print("   ⚙️ System Settings - Configure privacy and security parameters")
    print("\n🚀 To access the dashboard:")
    print("   streamlit run main_blockchain_enhanced.py")
    print("   Then navigate between pages using the sidebar!")

def main():
    """Main demo function"""
    print_banner()

    print("🚀 Initializing Enhanced IoT Blockchain System...")
    system = IoTPrivacySecuritySystem()

    print("✅ System initialized! Starting comprehensive demo...\n")

    # Wait a moment for some data to be generated
    print("📡 Generating initial sensor data...")
    time.sleep(5)

    # Run all demo sections
    demo_blockchain_visualization(system)
    demo_file_storage(system)
    demo_transaction_monitoring(system)
    demo_security_features(system)
    demo_dashboard_features()

    print("\n" + "="*80)
    print("🎯 ENHANCED DEMO COMPLETE!")
    print("="*80)
    print("✅ All enhanced features demonstrated:")
    print("   • Real blockchain transactions with visualization")
    print("   • File storage capabilities on blockchain")
    print("   • Comprehensive transaction monitoring")
    print("   • Advanced security and privacy features")
    print("   • Multi-page interactive dashboard")
    print("\n🌟 Your IoT Privacy & Security project is ready for academic evaluation!")
    print("📊 Access the full dashboard: streamlit run main_blockchain_enhanced.py")
    print("="*80)

if __name__ == "__main__":
    main()