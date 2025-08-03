import requests

url = "https://vrqvlim47f0206-8000.proxy.runpod.net/generate" 

data = {
    "prompt": "Explain the difference between a list and a tuple in Python.",
    "max_new_tokens": 150,
    "temperature": 0.7
}

response = requests.post(url, json=data)
print(response.json())
