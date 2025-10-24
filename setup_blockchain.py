#!/usr/bin/env python3
"""
Blockchain Setup Script for IoT Privacy & Security Project
==========================================================

This script helps set up the real blockchain environment for the project.
It can work with:
1. Local Ganache blockchain (recommended for development)
2. Ethereum testnet (Sepolia, Goerli)
3. Demo mode (simulated blockchain for testing)

Usage:
    python setup_blockchain.py --mode demo          # Demo mode
    python setup_blockchain.py --mode ganache       # Local Ganache
    python setup_blockchain.py --mode testnet       # Ethereum testnet
"""

import os
import sys
import json
import argparse
import subprocess
from typing import Dict, Optional

def install_ganache():
    """Install Ganache CLI for local blockchain"""
    print("📦 Installing Ganache CLI...")
    try:
        # Check if Node.js is installed
        subprocess.run(["node", "--version"], check=True, capture_output=True)

        # Install Ganache CLI
        subprocess.run(["npm", "install", "-g", "ganache-cli"], check=True)
        print("✅ Ganache CLI installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Node.js not found. Please install Node.js first:")
        print("   https://nodejs.org/")
        return False
    except FileNotFoundError:
        print("❌ npm not found. Please install Node.js first:")
        print("   https://nodejs.org/")
        return False

def start_ganache():
    """Start local Ganache blockchain"""
    print("🚀 Starting Ganache blockchain...")
    try:
        # Start Ganache with deterministic accounts
        cmd = [
            "ganache-cli",
            "--deterministic",
            "--accounts", "10",
            "--host", "127.0.0.1",
            "--port", "8545",
            "--networkId", "1337"
        ]

        print("Running:", " ".join(cmd))
        print("📡 Ganache will run on http://127.0.0.1:8545")
        print("💡 Keep this terminal open while using the system")
        print("🔑 Default accounts will be created with test ETH")

        # This will run in foreground - user needs to keep it open
        subprocess.run(cmd)

    except KeyboardInterrupt:
        print("\n🛑 Ganache stopped")
    except FileNotFoundError:
        print("❌ Ganache CLI not found. Run with --install flag first")

def compile_contract():
    """Compile the Solidity smart contract"""
    print("🔨 Compiling smart contract...")

    try:
        from solcx import compile_files, install_solc

        # Install Solidity compiler
        print("📦 Installing Solidity compiler...")
        install_solc('0.8.19')

        # Compile contract
        contract_file = "contracts/IoTDataStorage.sol"
        if not os.path.exists(contract_file):
            print(f"❌ Contract file not found: {contract_file}")
            return None

        compiled_sol = compile_files([contract_file])
        contract_interface = compiled_sol[f'{contract_file}:IoTDataStorage']

        # Save ABI and bytecode
        os.makedirs("contracts", exist_ok=True)

        with open("contracts/IoTDataStorage_abi.json", "w") as f:
            json.dump(contract_interface['abi'], f, indent=2)

        with open("contracts/IoTDataStorage_bytecode.json", "w") as f:
            json.dump({"bytecode": contract_interface['bin']}, f, indent=2)

        print("✅ Smart contract compiled successfully")
        return contract_interface

    except ImportError:
        print("❌ py-solc-x not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "py-solc-x"])
        print("✅ py-solc-x installed. Please run the script again.")
        return None
    except Exception as e:
        print(f"❌ Contract compilation failed: {e}")
        return None

def deploy_contract(private_key: str, blockchain_url: str = "http://127.0.0.1:8545"):
    """Deploy smart contract to blockchain"""
    print("🚀 Deploying smart contract...")

    try:
        from web3 import Web3
        from eth_account import Account

        # Connect to blockchain
        w3 = Web3(Web3.HTTPProvider(blockchain_url))

        if not w3.is_connected():
            print(f"❌ Cannot connect to blockchain at {blockchain_url}")
            return None

        print(f"✅ Connected to blockchain at {blockchain_url}")

        # Load compiled contract
        with open("contracts/IoTDataStorage_abi.json", "r") as f:
            contract_abi = json.load(f)

        with open("contracts/IoTDataStorage_bytecode.json", "r") as f:
            contract_bytecode = json.load(f)["bytecode"]

        # Setup account
        account = Account.from_key(private_key)

        # Check balance
        balance = w3.eth.get_balance(account.address)
        print(f"💰 Account balance: {w3.from_wei(balance, 'ether')} ETH")

        if balance == 0:
            print("❌ Account has no ETH. In Ganache, accounts start with test ETH automatically.")
            return None

        # Create contract
        contract = w3.eth.contract(abi=contract_abi, bytecode=contract_bytecode)

        # Build deployment transaction
        nonce = w3.eth.get_transaction_count(account.address)

        transaction = contract.constructor().build_transaction({
            'from': account.address,
            'nonce': nonce,
            'gas': 3000000,
            'gasPrice': w3.to_wei('20', 'gwei')
        })

        # Sign and send transaction
        signed_txn = w3.eth.account.sign_transaction(transaction, private_key)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)

        print(f"📝 Transaction sent: {tx_hash.hex()}")
        print("⏳ Waiting for confirmation...")

        # Wait for confirmation
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

        if receipt.status == 1:
            print(f"✅ Contract deployed successfully!")
            print(f"📋 Contract address: {receipt.contractAddress}")

            # Update config file
            update_config({
                'contract_address': receipt.contractAddress,
                'private_key': private_key,
                'blockchain_url': blockchain_url
            })

            return receipt.contractAddress
        else:
            print("❌ Contract deployment failed")
            return None

    except Exception as e:
        print(f"❌ Deployment error: {e}")
        return None

def update_config(updates: Dict):
    """Update configuration file with blockchain details"""
    config_file = "config.json"

    try:
        with open(config_file, "r") as f:
            config = json.load(f)

        config.update(updates)

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        print(f"✅ Configuration updated in {config_file}")

    except Exception as e:
        print(f"❌ Failed to update config: {e}")

def setup_demo_mode():
    """Setup demo mode configuration"""
    print("🎭 Setting up demo mode...")

    update_config({
        'blockchain_url': 'demo',
        'private_key': None,
        'contract_address': None
    })

    print("✅ Demo mode configured")
    print("📝 In demo mode, blockchain operations are simulated")
    print("🚀 You can now run: python main_blockchain.py")

def setup_ganache_mode():
    """Setup Ganache local blockchain mode"""
    print("⚙️ Setting up Ganache mode...")

    # Default Ganache account (deterministic)
    default_private_key = "0x4c0883a69102937d6231471b5dbb6204fe5129617082792ae468d01a3f362318"

    # Compile contract
    contract_interface = compile_contract()
    if not contract_interface:
        return False

    print("\n🔧 To complete setup:")
    print("1. Start Ganache in another terminal:")
    print("   python setup_blockchain.py --start-ganache")
    print("2. Then run deployment:")
    print("   python setup_blockchain.py --deploy")

    # Update config with Ganache settings
    update_config({
        'blockchain_url': 'http://127.0.0.1:8545',
        'private_key': default_private_key
    })

    return True

def main():
    parser = argparse.ArgumentParser(description="Setup blockchain for IoT Privacy & Security project")
    parser.add_argument("--mode", choices=["demo", "ganache", "testnet"],
                       default="demo", help="Blockchain mode")
    parser.add_argument("--install", action="store_true", help="Install Ganache CLI")
    parser.add_argument("--start-ganache", action="store_true", help="Start Ganache blockchain")
    parser.add_argument("--compile", action="store_true", help="Compile smart contract")
    parser.add_argument("--deploy", action="store_true", help="Deploy smart contract")
    parser.add_argument("--private-key", help="Private key for deployment")
    parser.add_argument("--url", default="http://127.0.0.1:8545", help="Blockchain URL")

    args = parser.parse_args()

    print("🏗️  IoT Privacy & Security Blockchain Setup")
    print("=" * 50)

    if args.install:
        install_ganache()
        return

    if args.start_ganache:
        start_ganache()
        return

    if args.compile:
        compile_contract()
        return

    if args.deploy:
        if not args.private_key:
            print("❌ Private key required for deployment")
            print("💡 For Ganache, use: --private-key 0x4c0883a69102937d6231471b5dbb6204fe5129617082792ae468d01a3f362318")
            return

        deploy_contract(args.private_key, args.url)
        return

    # Setup based on mode
    if args.mode == "demo":
        setup_demo_mode()
    elif args.mode == "ganache":
        if not setup_ganache_mode():
            return
    elif args.mode == "testnet":
        print("🌐 Testnet mode not implemented yet")
        print("💡 Use demo or ganache mode for now")
        return

    print("\n🚀 Setup complete! Next steps:")
    print("1. Run the system: python main_blockchain.py")
    print("2. Or start dashboard: streamlit run main_blockchain.py")
    print("\n📚 Project: IoT-Based Privacy-Preserving Data Management")
    print("           with Blockchain Integration and ML Intrusion Detection")

if __name__ == "__main__":
    main()