"""
NSL-KDD Dataset Download Script

Downloads the NSL-KDD dataset from the official source and validates the files.
"""

import urllib.request
import os
from pathlib import Path
import hashlib

# NSL-KDD dataset URLs
DATASET_URLS = {
    "train": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
    "test": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
    "train_20": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B_20Percent.txt",
    "test_21": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest-21.txt",
}

# Column names for NSL-KDD dataset (41 features + 2 labels)
COLUMN_NAMES = [
    # Basic features
    "duration", "protocol_type", "service", "flag",
    # Content features
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    # Count features
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    # Connection features (same host)
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    # Connection features (same service)
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    # Labels
    "attack_type", "difficulty_level"
]

# Attack type mappings
ATTACK_CATEGORIES = {
    "normal": "normal",
    # DoS attacks
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos", "udpstorm": "dos",
    "processtable": "dos", "worm": "dos", "mailbomb": "dos",
    # Probe attacks
    "satan": "probe", "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "mscan": "probe", "saint": "probe",
    # R2L attacks
    "guess_passwd": "r2l", "ftp_write": "r2l", "imap": "r2l",
    "phf": "r2l", "multihop": "r2l", "warezmaster": "r2l",
    "warezclient": "r2l", "spy": "r2l", "xlock": "r2l", "xsnoop": "r2l",
    "snmpguess": "r2l", "snmpgetattack": "r2l", "httptunnel": "r2l",
    "sendmail": "r2l", "named": "r2l",
    # U2R attacks
    "buffer_overflow": "u2r", "loadmodule": "u2r", "rootkit": "u2r",
    "perl": "u2r", "sqlattack": "u2r", "xterm": "u2r", "ps": "u2r"
}


def download_file(url: str, output_path: Path) -> bool:
    """
    Download a file from URL to output path.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"Saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def validate_file(file_path: Path) -> bool:
    """
    Validate that the downloaded file is not empty and has expected format.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        if not file_path.exists():
            print(f"File {file_path} does not exist")
            return False
            
        if file_path.stat().st_size == 0:
            print(f"File {file_path} is empty")
            return False
            
        # Check first line format
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            fields = first_line.split(',')
            if len(fields) < 41:
                print(f"File {file_path} has invalid format: expected at least 41 fields, got {len(fields)}")
                return False
                
        print(f"File {file_path} is valid")
        return True
    except Exception as e:
        print(f"Error validating {file_path}: {e}")
        return False


def download_nsl_kdd(output_dir: str = "ml/data/raw") -> bool:
    """
    Download all NSL-KDD dataset files.
    
    Args:
        output_dir: Directory to save downloaded files
        
    Returns:
        bool: True if all downloads successful, False otherwise
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    success = True
    for name, url in DATASET_URLS.items():
        file_path = output_path / f"KDD{name}.txt"
        
        # Skip if file already exists and is valid
        if file_path.exists() and validate_file(file_path):
            print(f"File {file_path} already exists and is valid, skipping...")
            continue
            
        # Download and validate
        if download_file(url, file_path):
            if not validate_file(file_path):
                success = False
        else:
            success = False
            
    return success


def get_data_statistics(file_path: Path) -> dict:
    """
    Get statistics about the dataset file.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        dict: Statistics including record count, attack distribution, etc.
    """
    try:
        record_count = 0
        attack_counts = {}
        
        with open(file_path, 'r') as f:
            for line in f:
                record_count += 1
                fields = line.strip().split(',')
                if len(fields) >= 42:
                    attack = fields[41].strip()
                    attack_counts[attack] = attack_counts.get(attack, 0) + 1
                    
        return {
            "file": str(file_path),
            "total_records": record_count,
            "attack_distribution": attack_counts
        }
    except Exception as e:
        print(f"Error getting statistics: {e}")
        return {}


if __name__ == "__main__":
    print("=" * 60)
    print("NSL-KDD Dataset Downloader")
    print("=" * 60)
    
    # Download datasets
    if download_nsl_kdd():
        print("\n✓ All datasets downloaded successfully!")
        
        # Print statistics
        print("\nDataset Statistics:")
        print("-" * 60)
        train_path = Path("ml/data/raw/KDDtrain.txt")
        if train_path.exists():
            stats = get_data_statistics(train_path)
            print(f"\nTraining Set: {stats.get('total_records', 0)} records")
            print("\nAttack Distribution:")
            for attack, count in sorted(stats.get('attack_distribution', {}).items()):
                print(f"  {attack}: {count}")
    else:
        print("\n✗ Some downloads failed. Please check errors above.")







