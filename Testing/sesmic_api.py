import requests
from obspy import read


# Define the parameters for a smaller query
base_url = 'https://geofon.gfz-potsdam.de/fdsnws/dataselect/1/query'
start_time = '2024-09-01T00:00:00'
end_time = '2024-09-01T00:00:01'  # Shorter time window

# Construct the request URL
request_url = f"{base_url}?starttime={start_time}&endtime={end_time}"
print(request_url)
# Make the request
response = requests.get(request_url, stream=True)

# Check if the request was successful
if response.status_code == 200:
    # Save the data to a file
    with open('seismic_data.mseed', 'wb') as f:
        print("Opened File")
        f.write(response.content)
        print("Wrote to and closed file")
    print("Data successfully downloaded.")
else:
    print(f"Failed to retrieve data: {response.status_code}")

print("Begin Reading")
# Read and process the data using ObsPy
st = read('seismic_data.mseed')
print(st)
