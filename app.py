from flask import Flask, render_template, request, jsonify, send_file
import os
import subprocess
import json
import tempfile
from werkzeug.utils import secure_filename
import sys

# Add the review 2 directory to Python path
REVIEW2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "review 2")
sys.path.append(REVIEW2_DIR)
from chatbot import maternal_chatbot  # Import the chatbot function directly

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'dat'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define directories
EHGPRETERM_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEW2_DIR = os.path.join(EHGPRETERM_DIR, "review 2")

# Define script paths
PREDICT_SCRIPT = os.path.join(REVIEW2_DIR, "preterm_prediction.py")
ACIDEMIA_DOCTOR_SCRIPT = os.path.join(REVIEW2_DIR, "acidemia_file_prediction.py")
EHG_PREDICTION_SCRIPT = os.path.join(REVIEW2_DIR, "ehg_preterm_prediction.py")
CHATBOT_SCRIPT = os.path.join(REVIEW2_DIR, "chatbot.py")
ACIDEMIA_PATIENT_SCRIPT = os.path.join(REVIEW2_DIR, "acidemia_prediction_form.py")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/doctor')
def doctor():
    return render_template('doctor.html')

@app.route('/patient')
def patient():
    return render_template('patient.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if not request.is_json:
            return jsonify({'error': 'Invalid request format'})
        
        message = request.json.get('message', '')
        if not message:
            return jsonify({'error': 'No message provided'})
        
        # Get response directly from the chatbot function
        response = maternal_chatbot(message)
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict/<user_type>/<prediction_type>', methods=['POST'])
def predict(user_type, prediction_type):
    try:
        script_path = None
        env = os.environ.copy()
        
        if user_type == 'doctor':
            if prediction_type == 'preterm':
                script_path = EHG_PREDICTION_SCRIPT
                # Handle file upload
                if 'ehg_file' not in request.files:
                    return jsonify({'error': 'No file uploaded'})
                file = request.files['ehg_file']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'})
                if not allowed_file(file.filename):
                    return jsonify({'error': 'Invalid file type. Please upload a .dat file containing EHG signals'})
                
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                env['EHG_FILE'] = filepath
                
            elif prediction_type == 'acidemia':
                script_path = ACIDEMIA_DOCTOR_SCRIPT
                # Handle file upload
                if 'ctg_file' not in request.files:
                    return jsonify({'error': 'No file uploaded'})
                file = request.files['ctg_file']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'})
                if not allowed_file(file.filename):
                    return jsonify({'error': 'Invalid file type. Please upload a .dat file containing CTG signals'})
                
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                env['CTG_FILE'] = filepath
            elif prediction_type == 'chatbot':
                return jsonify({'error': 'Please use the /chat endpoint for chatbot interactions'})
                
        elif user_type == 'patient':
            if prediction_type == 'preterm':
                script_path = PREDICT_SCRIPT
                
                # Get form data with proper validation
                form_data = {
                    'maternal_age': request.form.get('maternal_age_preterm', '').strip(),
                    'gestational_age': request.form.get('gestational_age_preterm', '').strip(),
                    'systolic_bp': request.form.get('systolic_bp', '').strip(),
                    'diastolic_bp': request.form.get('diastolic_bp', '').strip(),
                    'weight': request.form.get('weight', '').strip(),
                    'height': request.form.get('height', '').strip(),
                    'previous_pregnancies': request.form.get('previous_pregnancies', '0').strip(),
                    'diabetes': request.form.get('diabetes', '0').strip()
                }
                
                # Check for missing required fields
                missing_fields = [field for field, value in form_data.items() 
                                if not value and field not in ['previous_pregnancies', 'diabetes']]
                if missing_fields:
                    return jsonify({
                        'error': f"Missing required fields: {', '.join(missing_fields)}"
                    })
                
                try:
                    # Convert and validate numeric values
                    maternal_age = float(form_data['maternal_age'])
                    gestational_age = float(form_data['gestational_age'])
                    systolic_bp = float(form_data['systolic_bp'])
                    diastolic_bp = float(form_data['diastolic_bp'])
                    weight = float(form_data['weight'])
                    height = float(form_data['height'])
                    
                    # Calculate BMI
                    height_m = height / 100  # Convert cm to m
                    bmi = weight / (height_m * height_m)
                    
                    # Set environment variables with validated data
                    env.update({
                        'MATERNAL_AGE': str(maternal_age),
                        'GESTATIONAL_AGE': str(gestational_age),
                        'SYSTOLIC_BP': str(systolic_bp),
                        'DIASTOLIC_BP': str(diastolic_bp),
                        'BMI': str(bmi),
                        'WEIGHT': str(weight),
                        'HEIGHT': str(height),
                        'PREVIOUS_PREGNANCIES': form_data['previous_pregnancies'],
                        'DIABETES': form_data['diabetes']
                    })
                    
                except ValueError as e:
                    return jsonify({
                        'error': f'Invalid numeric value in form data: {str(e)}'
                    })
                
            elif prediction_type == 'acidemia':
                script_path = ACIDEMIA_PATIENT_SCRIPT
                
                # Get form data
                try:
                    env.update({
                        'MATERNAL_AGE': str(request.form.get('maternal_age', '')),
                        'GESTATIONAL_AGE': str(request.form.get('gestational_age', '')),
                        'GRAVIDA': str(request.form.get('gravida', '')),
                        'PARITY': str(request.form.get('parity', '')),
                        'APGAR_1': str(request.form.get('apgar_1', '0')),
                        'APGAR_5': str(request.form.get('apgar_5', '0')),
                        'PCO2': str(request.form.get('pco2', '0'))
                    })
                except (ValueError, TypeError) as e:
                    return jsonify({'error': f'Invalid form data: {str(e)}'})
                
            elif prediction_type == 'chatbot':
                return jsonify({'error': 'Please use the /chat endpoint for chatbot interactions'})
        
        if script_path is None:
            return jsonify({'error': 'Invalid prediction type or user type'})
            
        if not os.path.exists(script_path):
            return jsonify({'error': f'Prediction script not found: {os.path.basename(script_path)}'})
        
        # Run the prediction script with the environment variables
        result = subprocess.run(
            ["python", script_path], 
            capture_output=True, 
            text=True,
            encoding='utf-8',
            env=env
        )
        
        # Clean up uploaded file if it exists
        if 'EHG_FILE' in env and os.path.exists(env['EHG_FILE']):
            os.remove(env['EHG_FILE'])
        if 'CTG_FILE' in env and os.path.exists(env['CTG_FILE']):
            os.remove(env['CTG_FILE'])
        
        # Check for errors in the subprocess execution
        if result.returncode != 0:
            error_message = result.stderr.strip() if result.stderr else result.stdout.strip()
            if not error_message:
                error_message = "Unknown error occurred during prediction"
            print("Script error output:", error_message)
            return jsonify({'error': error_message})
        
        # Clean and format the output
        output = result.stdout.strip()
        if not output:
            return jsonify({'error': 'No prediction results received'})
        
        try:
            # If the output is JSON formatted, parse it
            if output.startswith('{'):
                data = json.loads(output)
                output = data.get('output', '')
            
            # Clean up the output
            cleaned_lines = []
            for line in output.split('\n'):
                line = line.strip()
                # Skip empty lines and unwanted content
                if line and not line.startswith('{"') and not line.endswith('"}'):
                    if '* Document all interventions' not in line and '"error": null' not in line:
                        cleaned_lines.append(line)
            
            cleaned_output = '\n'.join(cleaned_lines)
            
            # Remove any remaining JSON artifacts
            cleaned_output = cleaned_output.replace('\\n', '\n').strip('"')
            
            return jsonify({
                'output': cleaned_output,
                'error': None
            })
        except json.JSONDecodeError:
            # If not JSON, return the raw output
            return jsonify({
                'output': output,
                'error': None
            })
    
    except Exception as e:
        print("Exception in prediction endpoint:", str(e))
        # Clean up uploaded file in case of error
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Server error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True) 