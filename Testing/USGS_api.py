import requests

# Define the base URL for the API
base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Define the parameters for the query
params = {
    "format": "geojson",
    "starttime": "2024-09-07",
    "endtime": "2024-09-08",
    "minmagnitude": 5
}

# Send a GET request to the API
response = requests.get(base_url, params=params)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    data = response.json()  # Parse the response as JSON
    print("Number of earthquakes:", len(data['features']))
    for earthquake in data['features']:
        properties = earthquake['properties']
        #print(f"Place: {properties['place']}, Magnitude: {properties['mag']}")
        print(properties.keys())
        for key in properties.keys():
            print(f"{key} : {properties[key]}")
        break
else:
    print(f"Error: {response.status_code}")
