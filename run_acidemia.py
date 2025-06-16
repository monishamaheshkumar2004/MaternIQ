import os
import sys
import subprocess

def main():
    # Set TensorFlow environment variables to suppress warnings
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
    
    try:
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        check_script = os.path.join(current_dir, 'check.py')
        
        # Run check.py with the configured environment
        result = subprocess.run([sys.executable, check_script], 
                              capture_output=True, 
                              text=True,
                              encoding='utf-8')
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
            
        sys.exit(result.returncode)
        
    except Exception as e:
        print(f"Error running acidemia prediction: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 