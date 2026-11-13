
import os
from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)

api_key = os.getenv("GOOGLE_API_KEY")
model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

print("NEXUS AI - Gemini Interactions Test", flush=True)
print("Model:", model, flush=True)

try:
    client = genai.Client(api_key=api_key)

    print("Sending request...", flush=True)

    response = client.interactions.create(
        model=model,
        input="Reply with exactly: NEXUS AI CONNECTION SUCCESSFUL."
    )

    print("\nSUCCESS!")
    print(response.output_text)

except Exception as e:
    print("\nGEMINI ERROR:")
    print(type(e).__name__, str(e))
