from google import genai

# Use your API key here
client = genai.Client(api_key="")

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="what will be your accuracy if I use gemini1.5 flash for text to sql task? and will it be able to understand the schema of the database? Free to use?"
)

print(response.text)