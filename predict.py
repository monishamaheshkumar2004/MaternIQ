import os
import sys
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.preprocessing import StandardScaler

# Suppress scikit-learn version warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Get the directory containing the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def calculate_bmi(weight_kg, height_cm):
    """Calculate BMI from weight in kg and height in cm."""
    height_m = height_cm / 100
    return weight_kg / (height_m * height_m)

def preprocess_user_inputs(maternal_age, gestational_age, systolic_bp, diastolic_bp, 
                          weight, height, previous_pregnancies, diabetes):
    """Process user inputs into features."""
    try:
        # Convert inputs to appropriate types
        maternal_age = float(maternal_age)
        gestational_age = float(gestational_age)
        systolic_bp = float(systolic_bp)
        diastolic_bp = float(diastolic_bp)
        weight = float(weight)
        height = float(height)
        previous_pregnancies = int(previous_pregnancies)
        diabetes = int(diabetes)
        
        # Input validation
        if not (15 <= maternal_age <= 60):
            raise ValueError("Maternal age should be between 15 and 60 years")
        if not (20 <= gestational_age <= 45):
            raise ValueError("Gestational age should be between 20 and 45 weeks")
        if not (70 <= systolic_bp <= 200):
            raise ValueError("Systolic BP should be between 70 and 200 mmHg")
        if not (40 <= diastolic_bp <= 120):
            raise ValueError("Diastolic BP should be between 40 and 120 mmHg")
        if not (35 <= weight <= 200):
            raise ValueError("Weight should be between 35 and 200 kg")
        if not (140 <= height <= 220):
            raise ValueError("Height should be between 140 and 220 cm")
        if not (0 <= previous_pregnancies <= 15):
            raise ValueError("Number of previous pregnancies should be between 0 and 15")
        if diabetes not in [0, 1]:
            raise ValueError("Diabetes should be either 0 (No) or 1 (Yes)")
        
        # Calculate derived features
        bmi = calculate_bmi(weight, height)
        bp_ratio = systolic_bp / diastolic_bp if diastolic_bp > 0 else 0
        pulse_pressure = systolic_bp - diastolic_bp
        mean_arterial_pressure = (systolic_bp + (2 * diastolic_bp)) / 3
        
        # Create expanded feature vector with derived and interaction features
        features = [
            maternal_age,                    # Base feature 1
            gestational_age,                 # Base feature 2
            systolic_bp,                     # Base feature 3
            diastolic_bp,                    # Base feature 4
            bp_ratio,                        # Derived feature 1
            pulse_pressure,                  # Derived feature 2
            mean_arterial_pressure,          # Derived feature 3
            bmi,                            # Derived feature 4
            weight,                         # Base feature 5
            height,                         # Base feature 6
            previous_pregnancies,           # Base feature 7
            diabetes,                       # Base feature 8
            maternal_age * gestational_age,  # Interaction 1
            maternal_age * bmi,             # Interaction 2
            gestational_age * bmi,          # Interaction 3
            systolic_bp * diastolic_bp,     # Interaction 4
            maternal_age ** 2,              # Squared term 1
            gestational_age ** 2,           # Squared term 2
            bmi ** 2,                       # Squared term 3
            systolic_bp ** 2,               # Squared term 4
            diastolic_bp ** 2,              # Squared term 5
            np.log(maternal_age),           # Log transform 1
            np.log(gestational_age),        # Log transform 2
            np.log(bmi + 1),                # Log transform 3
            np.log(systolic_bp),            # Log transform 4
            np.log(diastolic_bp),           # Log transform 5
            np.sin(gestational_age * np.pi / 45),  # Cyclic feature 1
            np.cos(gestational_age * np.pi / 45),  # Cyclic feature 2
            maternal_age / gestational_age,  # Ratio feature 1
            weight / height,                # Ratio feature 2
            previous_pregnancies * diabetes, # Clinical interaction 1
            bmi * diabetes,                 # Clinical interaction 2
            pulse_pressure * diabetes        # Clinical interaction 3
        ]
        
        return np.array(features).reshape(1, -1)

    except Exception as e:
        print(f"Error processing user inputs: {str(e)}")
        return None

def format_prediction_result(prediction_prob):
    """Format the prediction result into a user-friendly message."""
    risk_level = prediction_prob[0][1]
    
    if risk_level >= 0.7:
        risk_category = "High"
        recommendation = [
            "1. Contact your healthcare provider immediately",
            "2. Monitor for signs of labor",
            "3. Consider hospital visit for evaluation",
            "4. Rest and avoid strenuous activities",
            "5. Prepare for possible early delivery"
        ]
    elif risk_level >= 0.4:
        risk_category = "Moderate"
        recommendation = [
            "1. Schedule follow-up with healthcare provider",
            "2. Monitor symptoms closely",
            "3. Get adequate rest",
            "4. Stay hydrated",
            "5. Avoid unnecessary physical strain"
        ]
    else:
        risk_category = "Low"
        recommendation = [
            "1. Continue regular prenatal care",
            "2. Maintain healthy lifestyle",
            "3. Stay active as advised by doctor",
            "4. Monitor baby's movements",
            "5. Keep scheduled appointments"
        ]
    
    result = [
        "Preterm Birth Risk Assessment",
        "-------------------------",
        f"Risk Category: {risk_category}",
        f"Risk Probability: {risk_level:.1%}",
        "",
        "Recommendations:",
        "---------------"
    ]
    result.extend(recommendation)
    
    # Add BMI information if available
    try:
        weight = float(os.environ.get('WEIGHT', 0))
        height = float(os.environ.get('HEIGHT', 0))
        if weight > 0 and height > 0:
            bmi = calculate_bmi(weight, height)
            result.extend([
                "",
                "Additional Information:",
                "--------------------",
                f"BMI: {bmi:.1f}",
                f"BMI Category: {get_bmi_category(bmi)}"
            ])
    except:
        pass
    
    return "\n".join(result)

def get_bmi_category(bmi):
    """Return BMI category based on BMI value."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def main():
    try:
        # Get user inputs from environment variables
        user_features = preprocess_user_inputs(
            os.environ.get('MATERNAL_AGE', ''),
            os.environ.get('GESTATIONAL_AGE', ''),
            os.environ.get('SYSTOLIC_BP', ''),
            os.environ.get('DIASTOLIC_BP', ''),
            os.environ.get('WEIGHT', ''),
            os.environ.get('HEIGHT', ''),
            os.environ.get('PREVIOUS_PREGNANCIES', ''),
            os.environ.get('DIABETES', '')
        )
        
        if user_features is None:
            print("Error: Invalid or missing user inputs")
            sys.exit(1)
        
        try:
            # Load the model using absolute path
            model_path = os.path.join(SCRIPT_DIR, "random_forest_best.pkl")
            if not os.path.exists(model_path):
                print(f"Error: Model file not found at {model_path}")
                sys.exit(1)
                
            model = joblib.load(model_path)
            
            # Standardize the features
            scaler = StandardScaler()
            user_features_scaled = scaler.fit_transform(user_features)
            
        # Make prediction
            prediction_prob = model.predict_proba(user_features_scaled)
            
            # Format and print results
            print(format_prediction_result(prediction_prob))
            
        except Exception as e:
            print(f"Error in model prediction: {str(e)}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error in input processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
