# Imports
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
import traceback  # Add this for detailed error tracking

# ------------------------
# 🧠 Extract data from .hea
# ------------------------
def parse_hea_file(file_path):
    data = {
        "maternal_age": np.nan,
        "gestational_age": np.nan,
        "gravida": np.nan,
        "parity": np.nan,
        "apgar_1": np.nan,
        "apgar_5": np.nan,
        "pco2": np.nan,
        "base_excess": np.nan,
        "base_deficit": np.nan,
        "pH": np.nan
    }
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                try:
                    if line.startswith("#Age"):
                        data["maternal_age"] = int(line.split()[1])
                    elif line.startswith("#Gest. weeks"):
                        data["gestational_age"] = int(line.split()[-1])
                    elif line.startswith("#Gravida"):
                        data["gravida"] = int(line.split()[1])
                    elif line.startswith("#Parity"):
                        data["parity"] = int(line.split()[1])
                    elif line.startswith("#Apgar 1 min"):
                        data["apgar_1"] = float(line.split()[3])
                    elif line.startswith("#Apgar 5 min"):
                        data["apgar_5"] = float(line.split()[3])
                    elif line.startswith("#pCO2"):
                        data["pco2"] = float(line.split()[1])
                    elif line.startswith("#Base excess"):
                        data["base_excess"] = float(line.split()[2])
                    elif line.startswith("#Base deficit"):
                        data["base_deficit"] = float(line.split()[2])
                    elif line.startswith("#pH"):
                        data["pH"] = float(line.split()[1])
                except (IndexError, ValueError) as e:
                    continue
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
    return data

# ------------------------
# 🧱 Create dataset
# ------------------------
def create_patient_dataset(hea_dir):
    records = []
    labels = []
    
    if not os.path.exists(hea_dir):
        raise ValueError(f"Dataset directory not found: {hea_dir}")
        
    print(f"Processing files in {hea_dir}...")
    file_count = 0
    
    for root, _, files in os.walk(hea_dir):
        for filename in files:
            if filename.endswith('.hea'):
                file_path = os.path.join(root, filename)
                data = parse_hea_file(file_path)
                
                # Check for required fields
                if np.isnan(data['pH']) or np.isnan(data['maternal_age']) or np.isnan(data['gestational_age']):
                    continue
                    
                label = 1 if data['pH'] < 7.2 else 0
                record = {k: data[k] for k in data if k != 'pH'}
                records.append(record)
                labels.append(label)
                file_count += 1
                
                if file_count % 100 == 0:
                    print(f"Processed {file_count} files...")
    
    print(f"Total files processed: {file_count}")
    
    if not records:
        raise ValueError("No valid records found in the dataset")
        
    df = pd.DataFrame(records)
    df['acidemia'] = labels
    
    print(f"Dataset created with {len(df)} records")
    print(f"Acidemia cases: {df['acidemia'].sum()} ({(df['acidemia'].sum()/len(df))*100:.1f}%)")
    
    return df

# ------------------------
# 👤 Get Patient Input from Environment Variables
# ------------------------
def get_patient_input_from_env():
    try:
        return {
            "maternal_age": int(os.environ.get('MATERNAL_AGE')),
            "gestational_age": int(os.environ.get('GESTATIONAL_AGE')),
            "gravida": int(os.environ.get('GRAVIDA')),
            "parity": int(os.environ.get('PARITY')),
            "apgar_1": float(os.environ.get('APGAR_1')) if os.environ.get('APGAR_1') else np.nan,
            "apgar_5": float(os.environ.get('APGAR_5')) if os.environ.get('APGAR_5') else np.nan,
            "pco2": float(os.environ.get('PCO2')) if os.environ.get('PCO2') else np.nan,
            "base_excess": float(os.environ.get('BASE_EXCESS')) if os.environ.get('BASE_EXCESS') else np.nan,
            "base_deficit": float(os.environ.get('BASE_DEFICIT')) if os.environ.get('BASE_DEFICIT') else np.nan
        }
    except (TypeError, ValueError) as e:
        print(f"Error: Invalid input data - {str(e)}")
        return None

# ------------------------
# 🔍 Predict
# ------------------------
def predict_acidemia(model, imputer, patient_data):
    try:
        df = pd.DataFrame([patient_data])
        df_imputed = imputer.transform(df)
        prediction = model.predict(df_imputed)[0]
        probabilities = model.predict_proba(df_imputed)[0]
        return prediction, probabilities
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, None

# ------------------------
# 🚀 Main execution
# ------------------------
if __name__ == "__main__":
    try:
        # Set up paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.dirname(current_dir)  # Go up one level to workspace root
        ctu_dir = os.path.join(workspace_dir, "ctu-chb-intrapartum-cardiotocography-database-1.0.0")
        model_path = os.path.join(current_dir, "neonatal_model.pkl")

        print("Starting acidemia prediction system...")
        print(f"Using CTU dataset from: {ctu_dir}")
        
        # Check if model already exists and is valid
        if os.path.exists(model_path):
            try:
                print("Loading existing model...")
                model, imputer = joblib.load(model_path)
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error loading existing model: {str(e)}")
                print("Will train a new model.")
                model = None
        else:
            print("No existing model found. Will train a new model.")
            model = None

        if model is None:
            # Create and train model with CTU dataset
            print("\nProcessing CTU dataset and training model...")
            try:
                df = create_patient_dataset(ctu_dir)
                
                X = df.drop(columns=['acidemia'])
                y = df['acidemia']

                # Impute missing values
                print("\nPreparing data...")
                imputer = SimpleImputer(strategy="mean")
                X_imputed = imputer.fit_transform(X)

                # Balance dataset
                print("Balancing dataset with SMOTE...")
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X_imputed, y)

                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

                # Train model
                print("\nTraining Random Forest model...")
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                # Evaluate model
                y_pred = model.predict(X_test)
                print("\nModel Performance:")
                print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred))

                # Save model
                print("\nSaving model...")
                joblib.dump((model, imputer), model_path)
                print(f"Model saved to: {model_path}")
            except Exception as e:
                print(f"Error during model training: {str(e)}")
                print("Detailed error:")
                print(traceback.format_exc())
                raise

        # Get patient input from environment variables
        print("\nProcessing patient input...")
        try:
            patient_data = get_patient_input_from_env()
            if patient_data is None:
                raise ValueError("Could not get valid patient data from input")
            
            print("\nReceived patient data:")
            for key, value in patient_data.items():
                print(f"{key}: {value}")
        except Exception as e:
            print(f"Error processing patient input: {str(e)}")
            print("Detailed error:")
            print(traceback.format_exc())
            raise

        # Make prediction
        print("\nMaking prediction...")
        try:
            prediction, probabilities = predict_acidemia(model, imputer, patient_data)
            if prediction is None:
                raise ValueError("Prediction failed")

            # Format and print results
            print("\nAcidemia Risk Assessment Results:")
            print("---------------------------------")
            print(f"Risk of Acidemia: {'High' if prediction == 1 else 'Low'}")
            print(f"Confidence: {probabilities[1]:.1%} chance of acidemia")
            
            # Print risk factors if high risk
            if prediction == 1:
                print("\nRisk Factors:")
                if patient_data['gestational_age'] < 37:
                    print("- Preterm pregnancy")
                if patient_data['maternal_age'] > 35:
                    print("- Advanced maternal age")
                if not np.isnan(patient_data['base_deficit']) and patient_data['base_deficit'] > 12:
                    print("- Elevated base deficit")
                if not np.isnan(patient_data['pco2']) and patient_data['pco2'] > 45:
                    print("- Elevated pCO2 levels")

            print("\nRecommendation:")
            if prediction == 1:
                print("Based on the assessment, close monitoring is recommended.")
                print("Please consult with your healthcare provider for proper medical advice.")
            else:
                print("Your risk factors are within normal range.")
                print("Continue with regular prenatal care and monitoring.")

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            print("Detailed error:")
            print(traceback.format_exc())
            raise

    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        print("\nDetailed error trace:")
        print(traceback.format_exc())
        exit(1)
