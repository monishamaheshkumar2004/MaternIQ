import os
from groq import Groq
import sys

# Set your Groq API key securely
client = Groq(api_key="")

def maternal_chatbot(prompt):
    try:
        # Create a system message that focuses on maternal and fetal health
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system", 
                    "content": "You are a specialized maternal health assistant. Focus on providing information about pregnancy, fetal health, preterm birth risks, and fetal acidemia. Always remind users to consult healthcare providers for medical advice."
                },
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192"
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

def main():
    try:
        # Get user message from environment variable or command line
        user_message = os.environ.get('USER_MESSAGE', '')
        
        if len(sys.argv) > 1:
            user_message = ' '.join(sys.argv[1:])
        
        if not user_message:
            print("No message received. Please provide a medical question.")
            return
        
        # Get and print response
        response = maternal_chatbot(user_message)
        print(response)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure UTF-8 encoding
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    main() 
