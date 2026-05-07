import yaml
from openai import OpenAI

# Open and read the YAML file
with open("openai_creds.yaml", "r") as file:
    openai_creds = yaml.safe_load(file)

# Access the credentials
ENDPOINT = openai_creds["openai"]["endpoint"]
API_KEY = openai_creds["openai"]["api_key"]
DEPLOYMENT_NAME = openai_creds["openai"]["deployment_name"]

# Initialize the client
client = OpenAI(base_url=ENDPOINT, api_key=API_KEY)

# Test sample completions
completion = client.chat.completions.create(
    model=DEPLOYMENT_NAME,
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ],
)

print(completion.choices[0].message.content)
