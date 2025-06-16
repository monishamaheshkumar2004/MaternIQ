import os
import subprocess

# Define directories
EHGPRETERM_DIR = r"C:\Users\Pradeepa\Desktop\ehg_preterm"
REVIEW2_DIR = os.path.join(EHGPRETERM_DIR, "review 2")

# Define script paths
PREDICT_SCRIPT = os.path.join(EHGPRETERM_DIR, "predict.py")
ACIDEMIA_DOCTOR_SCRIPT = os.path.join(EHGPRETERM_DIR, "check.py")  
PRETERM_PREDICTION_SCRIPT = os.path.join(REVIEW2_DIR, "preterm_prediction.py")
CHATBOT_SCRIPT = os.path.join(REVIEW2_DIR, "chatbot.py")
ACIDEMIA_PATIENT_SCRIPT = os.path.join(EHGPRETERM_DIR, "acidemia_csv.py")  

def main():
    print("\nWelcome to the EHG Prediction System!")
    
    while True:
        print("\nAre you a:")
        print("1. Doctor")
        print("2. Patient")
        print("3. Exit")
        
        user_type = input("Enter your choice (1/2/3): ").strip()

        if user_type == "1":
            doctor_menu()
        elif user_type == "2":
            patient_menu()
        elif user_type == "3":
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def doctor_menu():
    while True:
        print("\nDoctor Options:")
        print("1. Preterm Birth Prediction")
        print("2. Acidemia Prediction (Doctor)")
        print("3. Chatbot")
        print("4. Back to Main Menu")

        choice = input("Enter your choice (1/2/3/4): ").strip()

        if choice == "1":
            print("🔹 Running Preterm Birth Prediction (Doctor)...")
            subprocess.run(["python", PREDICT_SCRIPT])

        elif choice == "2":
            print("🔹 Running Acidemia Prediction (Doctor)...")
            subprocess.run(["python", ACIDEMIA_DOCTOR_SCRIPT])

        elif choice == "3":
            print("🔹 Running Chatbot...")
            subprocess.run(["python", CHATBOT_SCRIPT])

        elif choice == "4":
            return  # Go back to main menu

        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

def patient_menu():
    while True:
        print("\nPatient Options:")
        print("1. Preterm & C-section Prediction")
        print("2. Acidemia Prediction (Patient)")  # <-- New Option
        print("3. Chatbot")
        print("4. Back to Main Menu")

        choice = input("Enter your choice (1/2/3/4): ").strip()

        if choice == "1":
            print("🔹 Running Preterm & C-section Prediction...")
            subprocess.run(["python", PRETERM_PREDICTION_SCRIPT])

        elif choice == "2":
            print("🔹 Running Acidemia Prediction (Patient)...")
            subprocess.run(["python", ACIDEMIA_PATIENT_SCRIPT])

        elif choice == "3":
            print("🔹 Running Chatbot...")
            subprocess.run(["python", CHATBOT_SCRIPT])

        elif choice == "4":
            return  # Go back to main menu

        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

# Run the main function
if __name__ == "__main__":
    main()
