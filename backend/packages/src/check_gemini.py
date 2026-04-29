from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def gemini_call(query:str):
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=query
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    print(gemini_call("Hi gemini"))
