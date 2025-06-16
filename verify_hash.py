import hashlib
import os

# Function to compute SHA-256 hash of a file
def compute_sha256(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

# Saving the hash
def save_hash(file_path):
    hash_value = compute_sha256(file_path)
    hash_file = file_path + ".hash"
    with open(hash_file, "w") as f:
        f.write(hash_value)
    print(f"Hash saved to {hash_file}")

# Verifying the hash
def verify_hash(file_path):
    original_hash_file = file_path + ".hash"
    if not os.path.exists(original_hash_file):
        print("❌ Hash file not found!")
        return False

    with open(original_hash_file, "r") as f:
        original_hash = f.read().strip()

    current_hash = compute_sha256(file_path)

    if original_hash == current_hash:
        print("✅ Signal integrity verified! No tampering.")
        return True
    else:
        print("❌ Tampering detected!")
        return False

# Example Usage
file_to_send = r"C:\Users\Pradeepa\Desktop\ehg_preterm\ctu-chb-intrapartum-cardiotocography-database-1.0.0\1004.dat"

# On sender side (lab)
save_hash(file_to_send)

# On receiver side (doctor or patient)
verify_hash(file_to_send)
