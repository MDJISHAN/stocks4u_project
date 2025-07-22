import requests

url = "https://webapi.niftytrader.in/webapi/Resource/bulk-deal-data"
headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    try:
        data = response.json()
        print(data)  # Print the dictionary directly
    except Exception as e:
        print("Error parsing JSON:", e)
else:
    print(f"Failed to fetch data: {response.status_code}")
