from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv(".env")   
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
if DEEPSEEK_API_KEY is None:
    raise ValueError("DEEPSEEK_API_KEY environment variable not set.")
DEEPSEEK_URL = os.getenv('DEEPSEEK_URL')
if DEEPSEEK_URL is None:
    raise ValueError("DEEPSEEK_URL environment variable not set.")

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

#Loop to prompt user for input and get response from the model
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat.")
        break

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": user_input},
        ],
        stream=False
    )
    
    # Print the model's response
    print("Assistant:", end=" ")
    print(response.choices[0].message.content)