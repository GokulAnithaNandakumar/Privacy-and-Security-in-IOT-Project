import hashlib
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from dataclasses import dataclass
import uuid

@dataclass
class Transaction:
    """Represents a single IoT sensor reading transaction"""
    transaction_id: str
    timestamp: float
    sensor_id: str
    room: str
    sensor_type: str
    value: Any
    signature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary"""
        return {
            'transaction_id': self.transaction_id,
            'timestamp': self.timestamp,
            'sensor_id': self.sensor_id,
            'room': self.room,
            'sensor_type': self.sensor_type,
            'value': self.value,
            'signature': self.signature
        }

    def calculate_hash(self) -> str:
        """Calculate hash of the transaction"""
        transaction_string = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(transaction_string.encode()).hexdigest()


@dataclass
class Block:
    """Represents a block in the blockchain"""
    index: int
    timestamp: float
    transactions: List[Transaction]
    previous_hash: str
    nonce: int = 0
    hash: Optional[str] = None

    def calculate_hash(self) -> str:
        """Calculate hash of the block"""
        block_dict = {
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }
        block_string = json.dumps(block_dict, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine_block(self, difficulty: int = 4) -> None:
        """Mine the block using proof of work"""
        target = "0" * difficulty
        start_time = time.time()

        while not self.hash or not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.calculate_hash()

            # Print progress every 10000 attempts
            if self.nonce % 10000 == 0:
                print(f"Mining block {self.index}: nonce = {self.nonce}")

        mining_time = time.time() - start_time
        print(f"Block {self.index} mined successfully! Hash: {self.hash}")
        print(f"Mining time: {mining_time:.2f} seconds")

    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary"""
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'hash': self.hash
        }


class IoTBlockchain:
    """Lightweight blockchain for IoT sensor data"""

    def __init__(self, difficulty: int = 4, max_transactions_per_block: int = 10):
        """
        Initialize the blockchain

        Args:
            difficulty: Mining difficulty (number of leading zeros)
            max_transactions_per_block: Maximum transactions per block
        """
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.difficulty = difficulty
        self.max_transactions_per_block = max_transactions_per_block
        self.mining_reward = 1.0

        # Create genesis block
        self.create_genesis_block()

    def create_genesis_block(self) -> None:
        """Create the first block in the chain"""
        genesis_transaction = Transaction(
            transaction_id=str(uuid.uuid4()),
            timestamp=time.time(),
            sensor_id="genesis",
            room="genesis",
            sensor_type="genesis",
            value="Genesis Block"
        )

        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            transactions=[genesis_transaction],
            previous_hash="0"
        )

        genesis_block.hash = genesis_block.calculate_hash()
        self.chain.append(genesis_block)
        print("Genesis block created")

    def get_latest_block(self) -> Block:
        """Get the latest block in the chain"""
        return self.chain[-1]

    def add_transaction(self, transaction: Transaction) -> None:
        """Add a transaction to the pending transactions"""
        self.pending_transactions.append(transaction)

        # Auto-mine block when enough transactions are collected
        if len(self.pending_transactions) >= self.max_transactions_per_block:
            self.mine_pending_transactions()

    def mine_pending_transactions(self) -> None:
        """Mine a new block with pending transactions"""
        if not self.pending_transactions:
            print("No pending transactions to mine")
            return

        # Create new block
        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            transactions=self.pending_transactions.copy(),
            previous_hash=self.get_latest_block().hash
        )

        # Mine the block
        print(f"Mining block {new_block.index} with {len(new_block.transactions)} transactions...")
        new_block.mine_block(self.difficulty)

        # Add to chain and clear pending transactions
        self.chain.append(new_block)
        self.pending_transactions = []

        print(f"Block {new_block.index} added to chain")

    def create_transaction_from_sensor_data(self, sensor_reading: Dict[str, Any]) -> Transaction:
        """Create a transaction from sensor reading"""
        return Transaction(
            transaction_id=str(uuid.uuid4()),
            timestamp=time.time(),
            sensor_id=sensor_reading.get('sensor_id', 'unknown'),
            room=sensor_reading.get('room', 'unknown'),
            sensor_type=sensor_reading.get('sensor_type', 'unknown'),
            value=sensor_reading.get('value', 0)
        )

    def add_sensor_data_batch(self, sensor_data: pd.DataFrame) -> None:
        """Add multiple sensor readings to the blockchain"""
        print(f"Adding {len(sensor_data)} sensor readings to blockchain...")

        for _, reading in sensor_data.iterrows():
            transaction = self.create_transaction_from_sensor_data(reading.to_dict())
            self.add_transaction(transaction)

        # Mine any remaining pending transactions
        if self.pending_transactions:
            self.mine_pending_transactions()

    def validate_chain(self) -> bool:
        """Validate the entire blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            # Check if current block's hash is valid
            if current_block.hash != current_block.calculate_hash():
                print(f"Invalid hash at block {i}")
                return False

            # Check if current block points to previous block
            if current_block.previous_hash != previous_block.hash:
                print(f"Invalid previous hash at block {i}")
                return False

        return True

    def get_blockchain_stats(self) -> Dict[str, Any]:
        """Get statistics about the blockchain"""
        total_transactions = sum(len(block.transactions) for block in self.chain)

        stats = {
            'total_blocks': len(self.chain),
            'total_transactions': total_transactions,
            'pending_transactions': len(self.pending_transactions),
            'blockchain_size_mb': self.calculate_blockchain_size(),
            'is_valid': self.validate_chain(),
            'latest_block_hash': self.get_latest_block().hash,
            'average_transactions_per_block': total_transactions / len(self.chain) if self.chain else 0
        }

        return stats

    def calculate_blockchain_size(self) -> float:
        """Calculate approximate size of blockchain in MB"""
        blockchain_json = json.dumps([block.to_dict() for block in self.chain])
        size_bytes = len(blockchain_json.encode('utf-8'))
        return size_bytes / (1024 * 1024)  # Convert to MB

    def get_transactions_by_sensor(self, sensor_id: str) -> List[Transaction]:
        """Get all transactions for a specific sensor"""
        transactions = []
        for block in self.chain:
            for transaction in block.transactions:
                if transaction.sensor_id == sensor_id:
                    transactions.append(transaction)
        return transactions

    def get_transactions_by_room(self, room: str) -> List[Transaction]:
        """Get all transactions for a specific room"""
        transactions = []
        for block in self.chain:
            for transaction in block.transactions:
                if transaction.room == room:
                    transactions.append(transaction)
        return transactions

    def get_transactions_by_time_range(self, start_time: float, end_time: float) -> List[Transaction]:
        """Get transactions within a time range"""
        transactions = []
        for block in self.chain:
            for transaction in block.transactions:
                if start_time <= transaction.timestamp <= end_time:
                    transactions.append(transaction)
        return transactions

    def export_blockchain(self, filepath: str) -> None:
        """Export blockchain to JSON file"""
        blockchain_data = {
            'blockchain': [block.to_dict() for block in self.chain],
            'pending_transactions': [tx.to_dict() for tx in self.pending_transactions],
            'stats': self.get_blockchain_stats(),
            'export_timestamp': time.time()
        }

        with open(filepath, 'w') as f:
            json.dump(blockchain_data, f, indent=2)

        print(f"Blockchain exported to {filepath}")

    def import_blockchain(self, filepath: str) -> None:
        """Import blockchain from JSON file"""
        with open(filepath, 'r') as f:
            blockchain_data = json.load(f)

        # Reconstruct blockchain
        self.chain = []
        for block_data in blockchain_data['blockchain']:
            transactions = []
            for tx_data in block_data['transactions']:
                transaction = Transaction(**tx_data)
                transactions.append(transaction)

            block = Block(
                index=block_data['index'],
                timestamp=block_data['timestamp'],
                transactions=transactions,
                previous_hash=block_data['previous_hash'],
                nonce=block_data['nonce'],
                hash=block_data['hash']
            )
            self.chain.append(block)

        # Reconstruct pending transactions
        self.pending_transactions = []
        for tx_data in blockchain_data.get('pending_transactions', []):
            transaction = Transaction(**tx_data)
            self.pending_transactions.append(transaction)

        print(f"Blockchain imported from {filepath}")
        print(f"Loaded {len(self.chain)} blocks with {sum(len(b.transactions) for b in self.chain)} transactions")

    def get_blockchain_dataframe(self) -> pd.DataFrame:
        """Convert blockchain to pandas DataFrame for analysis"""
        rows = []
        for block in self.chain:
            for transaction in block.transactions:
                row = {
                    'block_index': block.index,
                    'block_hash': block.hash,
                    'block_timestamp': block.timestamp,
                    'transaction_id': transaction.transaction_id,
                    'transaction_timestamp': transaction.timestamp,
                    'sensor_id': transaction.sensor_id,
                    'room': transaction.room,
                    'sensor_type': transaction.sensor_type,
                    'value': transaction.value
                }
                rows.append(row)

        return pd.DataFrame(rows)


class BlockchainNetwork:
    """Simulate a network of blockchain nodes"""

    def __init__(self, num_nodes: int = 3):
        """Initialize network with multiple nodes"""
        self.nodes = {}
        self.num_nodes = num_nodes

        for i in range(num_nodes):
            node_id = f"node_{i+1}"
            self.nodes[node_id] = IoTBlockchain(difficulty=2)  # Lower difficulty for demo

    def broadcast_transaction(self, transaction: Transaction) -> None:
        """Broadcast transaction to all nodes"""
        for node_id, blockchain in self.nodes.items():
            blockchain.add_transaction(transaction)

    def sync_nodes(self) -> None:
        """Synchronize all nodes to the longest valid chain"""
        # Find the longest valid chain
        longest_chain = []
        longest_length = 0

        for node_id, blockchain in self.nodes.items():
            if blockchain.validate_chain() and len(blockchain.chain) > longest_length:
                longest_chain = blockchain.chain
                longest_length = len(blockchain.chain)

        # Update all nodes with the longest chain
        if longest_chain:
            for node_id, blockchain in self.nodes.items():
                blockchain.chain = longest_chain.copy()
                print(f"Node {node_id} synchronized")

    def get_network_stats(self) -> Dict[str, Any]:
        """Get statistics about the network"""
        stats = {}
        for node_id, blockchain in self.nodes.items():
            stats[node_id] = blockchain.get_blockchain_stats()

        return stats


if __name__ == "__main__":
    # Example usage
    from ..sensors.sensor_simulator import IoTSensorSimulator

    # Generate sample sensor data
    simulator = IoTSensorSimulator(num_rooms=2)
    sensor_data = simulator.generate_full_dataset(duration_hours=2)  # Small dataset for demo

    # Create blockchain
    blockchain = IoTBlockchain(difficulty=2, max_transactions_per_block=5)

    # Add sensor data to blockchain
    blockchain.add_sensor_data_batch(sensor_data.head(20))  # Add first 20 readings

    # Print blockchain stats
    stats = blockchain.get_blockchain_stats()
    print(f"Blockchain stats: {stats}")

    # Validate blockchain
    is_valid = blockchain.validate_chain()
    print(f"Blockchain is valid: {is_valid}")

    # Export blockchain
    blockchain.export_blockchain("../data/iot_blockchain.json")
