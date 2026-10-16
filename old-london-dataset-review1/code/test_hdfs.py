import requests

url = "http://localhost:9870/webhdfs/v1/data/raw/old-london-dataset-review1/processed/energy_daily?op=LISTSTATUS"

r = requests.get(url)

print("HTTP Status:", r.status_code)
print(r.text[:1000])
