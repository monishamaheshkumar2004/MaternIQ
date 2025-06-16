import os

# Set dataset path
dataset_path = "C:/Users/Pradeepa/Desktop/ehg_preterm/term-preterm-ehg-database-1.0.1/tpehgdb"

# Initialize counters
term_count = 0
preterm_count = 0
missing_gestation = []

# Function to extract gestation from .hea file
def extract_gestation(hea_file):
    try:
        with open(hea_file, 'r') as file:
            for line in file:
                # Look for the word "Gestation" in the line
                if "Gestation" in line:
                    parts = line.strip().split()
                    for part in parts:
                        try:
                            return float(part)  # Convert the first valid number
                        except ValueError:
                            continue
        return None  # No valid number found
    except Exception as e:
        print(f"Error reading {hea_file}: {e}")
        return None

# Process all .hea files
for file in os.listdir(dataset_path):
    if file.endswith(".hea"):
        file_path = os.path.join(dataset_path, file)
        gestation = extract_gestation(file_path)

        if gestation is None:
            missing_gestation.append(file)  # Track files with missing gestation
        elif gestation >= 37:
            term_count += 1  # TERM case
        else:
            preterm_count += 1  # PRETERM case

# Print results
print("\n **Gestation Analysis from .hea Files**")
print(f"TERM Cases (≥37 weeks): {term_count}")
print(f"PRETERM Cases (<37 weeks): {preterm_count}")

# Print files where gestation is missing
if missing_gestation:
    print("\n **Files Missing Gestation Information:**")
    for filename in missing_gestation:
        print(f"- {filename}")
else:
    print("\n No missing gestation values detected!")
