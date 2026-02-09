from google import genai

# Explicit API key injected per user request.
API_KEY = "YOUR OWN API KEY"

client = genai.Client(api_key=API_KEY)

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Explain how AI works in a few words"
)
print(response.text)
