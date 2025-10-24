import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple
import json

class IoTSensorSimulator:
    """
    Simulates various IoT sensors generating time-series data
    """

    def __init__(self, num_rooms: int = 5, start_time: datetime = None):
        self.num_rooms = num_rooms
        self.start_time = start_time or datetime.now() - timedelta(days=7)
        self.sensor_types = ['motion', 'light', 'temperature', 'humidity', 'air_quality']
        self.rooms = [f"Room_{i+1}" for i in range(num_rooms)]

    def generate_motion_data(self, room: str, duration_hours: int = 24) -> pd.DataFrame:
        """Generate motion sensor data (binary: 0/1)"""
        timestamps = pd.date_range(
            start=self.start_time,
            periods=duration_hours * 60,  # Every minute
            freq='1min'
        )

        # Simulate realistic motion patterns
        motion_data = []
        for ts in timestamps:
            hour = ts.hour
            # Higher probability of motion during day (7-22), lower at night
            if 7 <= hour <= 22:
                motion_prob = 0.3  # 30% chance of motion during day
            else:
                motion_prob = 0.05  # 5% chance during night

            # Add some randomness for intrusions
            if random.random() < 0.001:  # 0.1% chance of unusual activity
                motion = 1
            else:
                motion = 1 if random.random() < motion_prob else 0

            motion_data.append({
                'timestamp': ts,
                'room': room,
                'sensor_type': 'motion',
                'value': motion,
                'sensor_id': f"{room}_motion_001"
            })

        return pd.DataFrame(motion_data)

    def generate_light_data(self, room: str, duration_hours: int = 24) -> pd.DataFrame:
        """Generate light sensor data (lux levels)"""
        timestamps = pd.date_range(
            start=self.start_time,
            periods=duration_hours * 12,  # Every 5 minutes
            freq='5min'
        )

        light_data = []
        for ts in timestamps:
            hour = ts.hour

            # Simulate natural light patterns
            if 6 <= hour <= 18:  # Daytime
                base_light = 300 + 200 * np.sin((hour - 6) * np.pi / 12)
            else:  # Nighttime
                base_light = 50 + random.uniform(-20, 20)

            # Add noise and occasional artificial light
            noise = random.uniform(-50, 50)
            artificial_light = 200 if random.random() < 0.2 else 0

            light_value = max(0, base_light + noise + artificial_light)

            light_data.append({
                'timestamp': ts,
                'room': room,
                'sensor_type': 'light',
                'value': round(light_value, 2),
                'sensor_id': f"{room}_light_001"
            })

        return pd.DataFrame(light_data)

    def generate_temperature_data(self, room: str, duration_hours: int = 24) -> pd.DataFrame:
        """Generate temperature sensor data (Celsius)"""
        timestamps = pd.date_range(
            start=self.start_time,
            periods=duration_hours * 4,  # Every 15 minutes
            freq='15min'
        )

        temp_data = []
        base_temp = random.uniform(20, 24)  # Base room temperature

        for i, ts in enumerate(timestamps):
            hour = ts.hour

            # Daily temperature variation
            daily_variation = 2 * np.sin((hour - 6) * np.pi / 12)

            # Gradual drift over time
            drift = (i / len(timestamps)) * random.uniform(-1, 1)

            # Random noise
            noise = random.uniform(-0.5, 0.5)

            # Occasional heating/cooling events
            hvac_effect = 0
            if random.random() < 0.1:  # 10% chance of HVAC activity
                hvac_effect = random.uniform(-2, 2)

            temperature = base_temp + daily_variation + drift + noise + hvac_effect

            temp_data.append({
                'timestamp': ts,
                'room': room,
                'sensor_type': 'temperature',
                'value': round(temperature, 2),
                'sensor_id': f"{room}_temp_001"
            })

        return pd.DataFrame(temp_data)

    def generate_humidity_data(self, room: str, duration_hours: int = 24) -> pd.DataFrame:
        """Generate humidity sensor data (percentage)"""
        timestamps = pd.date_range(
            start=self.start_time,
            periods=duration_hours * 4,  # Every 15 minutes
            freq='15min'
        )

        humidity_data = []
        base_humidity = random.uniform(40, 60)  # Base humidity

        for ts in timestamps:
            hour = ts.hour

            # Daily humidity variation (inverse of temperature)
            daily_variation = -3 * np.sin((hour - 6) * np.pi / 12)

            # Random noise
            noise = random.uniform(-2, 2)

            # Occasional spikes (showers, cooking, etc.)
            spike = 0
            if random.random() < 0.05:  # 5% chance of humidity spike
                spike = random.uniform(10, 20)

            humidity = np.clip(base_humidity + daily_variation + noise + spike, 0, 100)

            humidity_data.append({
                'timestamp': ts,
                'room': room,
                'sensor_type': 'humidity',
                'value': round(humidity, 2),
                'sensor_id': f"{room}_humidity_001"
            })

        return pd.DataFrame(humidity_data)

    def generate_air_quality_data(self, room: str, duration_hours: int = 24) -> pd.DataFrame:
        """Generate air quality sensor data (PM2.5 levels)"""
        timestamps = pd.date_range(
            start=self.start_time,
            periods=duration_hours * 2,  # Every 30 minutes
            freq='30min'
        )

        air_quality_data = []
        base_pm25 = random.uniform(10, 30)  # Base PM2.5 level

        for ts in timestamps:
            hour = ts.hour

            # Higher pollution during certain hours
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                pollution_factor = 1.5
            else:
                pollution_factor = 1.0

            # Random variation
            noise = random.uniform(-5, 5)

            # Occasional pollution events
            event = 0
            if random.random() < 0.02:  # 2% chance of pollution event
                event = random.uniform(20, 50)

            pm25_level = max(0, base_pm25 * pollution_factor + noise + event)

            air_quality_data.append({
                'timestamp': ts,
                'room': room,
                'sensor_type': 'air_quality',
                'value': round(pm25_level, 2),
                'sensor_id': f"{room}_air_001"
            })

        return pd.DataFrame(air_quality_data)

    def simulate_intrusion(self, data: pd.DataFrame, intrusion_probability: float = 0.001) -> pd.DataFrame:
        """Inject anomalous readings to simulate intrusions"""
        intrusion_data = data.copy()

        for i in range(len(intrusion_data)):
            if random.random() < intrusion_probability:
                sensor_type = intrusion_data.iloc[i]['sensor_type']

                if sensor_type == 'motion':
                    # Unexpected motion at night
                    hour = intrusion_data.iloc[i]['timestamp'].hour
                    if 0 <= hour <= 6:
                        intrusion_data.iloc[i, intrusion_data.columns.get_loc('value')] = 1

                elif sensor_type == 'light':
                    # Sudden light changes
                    intrusion_data.iloc[i, intrusion_data.columns.get_loc('value')] *= random.uniform(2, 5)

                elif sensor_type == 'temperature':
                    # Unusual temperature spikes
                    intrusion_data.iloc[i, intrusion_data.columns.get_loc('value')] += random.uniform(5, 10)

        return intrusion_data

    def generate_full_dataset(self, duration_hours: int = 168) -> pd.DataFrame:
        """Generate complete dataset for all rooms and sensors"""
        all_data = []

        for room in self.rooms:
            print(f"Generating data for {room}...")

            # Generate data for each sensor type
            motion_data = self.generate_motion_data(room, duration_hours)
            light_data = self.generate_light_data(room, duration_hours)
            temp_data = self.generate_temperature_data(room, duration_hours)
            humidity_data = self.generate_humidity_data(room, duration_hours)
            air_quality_data = self.generate_air_quality_data(room, duration_hours)

            # Combine all sensor data for this room
            room_data = pd.concat([
                motion_data, light_data, temp_data,
                humidity_data, air_quality_data
            ], ignore_index=True)

            # Simulate intrusions
            room_data = self.simulate_intrusion(room_data)

            all_data.append(room_data)

        # Combine all rooms data
        complete_dataset = pd.concat(all_data, ignore_index=True)
        complete_dataset = complete_dataset.sort_values('timestamp').reset_index(drop=True)

        return complete_dataset

    def save_dataset(self, dataset: pd.DataFrame, filepath: str):
        """Save dataset to file"""
        dataset.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")

    def get_sensor_stats(self, dataset: pd.DataFrame) -> Dict:
        """Get basic statistics about the generated dataset"""
        stats = {
            'total_readings': len(dataset),
            'rooms': dataset['room'].nunique(),
            'sensor_types': dataset['sensor_type'].nunique(),
            'time_range': {
                'start': dataset['timestamp'].min(),
                'end': dataset['timestamp'].max()
            },
            'readings_per_sensor': dataset['sensor_type'].value_counts().to_dict()
        }
        return stats


if __name__ == "__main__":
    # Example usage
    simulator = IoTSensorSimulator(num_rooms=3)
    dataset = simulator.generate_full_dataset(duration_hours=24)

    print("Dataset generated successfully!")
    print(f"Total readings: {len(dataset)}")
    print(f"Sensor types: {dataset['sensor_type'].unique()}")
    print(f"Rooms: {dataset['room'].unique()}")

    # Save dataset
    simulator.save_dataset(dataset, "../data/iot_sensor_data.csv")
