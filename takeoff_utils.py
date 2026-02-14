# %%
import geojson
import geopandas as gpd
import requests


def fetch_takeoffs_norway(limit=None):
    # docs
    # https://www.paraglidingearth.com/api/
    # URL of the GeoJSON endpoint
    geo_json_query = "http://www.paraglidingearth.com/api/geojson/getCountrySites.php"
    # Add query iso=578
    # Send a GET request to the GeoJSON endpoint
    params = {"iso": "NO"}
    if limit:
        params["limit"] = limit

    # Send a GET request to the GeoJSON endpoint with the query parameter
    response = requests.get(geo_json_query, params=params)
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the response as a GeoJSON object
        data = geojson.loads(response.text)
        # Print the GeoJSON data
    else:
        print(f"Error: Unable to fetch data (Status code: {response.status_code})")
        return None

    #
    """
    print(data)
    {"features": [{"geometry": {"coordinates": [10.463, 59.9396], "type": "Point"}, "id": "6603", "properties": {"E": "0", "N": "0", "NE": "0", "NW": "0", "S": "0", "SE": "0", "SW": "0", "W": "0", "countryCode": "no", "ffvl_site_id": "0", "flatland": "0", "hanggliding": "0", "landing_lat": "", "landing_lng": "", "landing_parking_lat": "", "landing_parking_lng": "", "last_edit": "", "name": "Solfjellstua", "paragliding": "1", "pge_link": "http://www.paraglidingearth.com/?site=6603", "pge_site_id": "6603", "place": "paragliding takeoff", "soaring": "0", "takeoff_altitude": "314", "takeoff_description": "", "takeoff_parking_lat": "", "takeoff_parking_lng": "", "thermals": "0", "winch": "0", "xc": "0"}, "type": "Feature"}, {"geometry": {"coordinates": [9.17167, 62.0494], "type": "Point"}, "id": "6715", "properties": {"E": "0", "N": "0", "NE": "0", "NW": "0", "S": "2", "SE": "0", "SW": "2", "W": "0", "countryCode": "no", "ffvl_site_id": "0", "flatland": "0", "hanggliding": "0", "landing_lat": "", "landing_lng": "", "landing_parking_lat": "", "landing_parking_lng": "", "last_edit": "", "name": "Engjekollen - Dombes", "paragliding": "1", "pge_link": "http://www.paraglidingearth.com/?site=6715", "pge_site_id": "6715", "place": "paragliding takeoff", "soaring": "0", "takeoff_altitude": "1007", "takeoff_description": "", "takeoff_parking_lat": "", "takeoff_parking_lng": "", "thermals": "0", "winch": "0", "xc": "0"}, "type": "Feature"}], "type": "FeatureCollection"}
    """
    # %%
    gdf = gpd.GeoDataFrame.from_features(data["features"])
    # records = [
    #     {
    #         "latitude": feature["geometry"]["coordinates"][1],
    #         "longitude": feature["geometry"]["coordinates"][0],
    #         "name": feature["properties"]["name"],
    #         "url" : feature["properties"]["pge_link"],
    #     }
    #     for feature in data['features']
    # ]

    # # Create a Polars DataFrame from the extracted records
    # df = pl.DataFrame(records)
    return gdf


if __name__ == "__main__":
    # Fetch the data and create a DataFram
    df_takeoffs = fetch_takeoffs_norway()
