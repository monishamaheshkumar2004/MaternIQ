import os

def list_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            print(os.path.join(root, file))

# Replace 'your_directory_path' with the actual directory path
list_files('your_directory_path')
