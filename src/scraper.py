import requests
import json
import pandas as pd
from time import time, strftime, gmtime

class Scraper:
    def __init__(self):
        self.raw_url = None
        self.folder = None
        self.json = None
        self.df = None
        
    def get_json(self, *args):
        url = self.raw_url.format(*args)
        response = requests.get(url)
        
        if response.status_code == 200:
            print("Request OK")
            self.json = response.json()
            return True
        else:
            print(f'Error: Request failed with status code {response.status_code}')
            return False
        
    def to_dataframe(self):
        pass
        
    def store(self, filename):
        self.df.dropna(thresh=3, inplace=True)
        path = "data" + "\\" + self.folder + "\\" + filename
        self.df.to_csv(path, index=False)
        
    def scrape_and_store(self, filename, *args):
        if self.get_json(*args):
            self.to_dataframe()
            self.store(filename)
            
            
class TimeSeriesScraper(Scraper):
    def mass_scrape(self, df, start, end):
        pass
            
            
class CountryScraper(Scraper):
    def __init__(self):
        super().__init__()
        self.raw_url = "http://api.geonames.org/countryInfoJSON?username=LucasLocker222"
        self.folder = "countries"
        
    def to_dataframe(self):
        path = "data" + "\\" + self.folder + "\\" + 'countries.json'
        with open(path, 'w') as json_file:
            json.dump(self.json["geonames"], json_file, indent=4)  
        self.df = pd.read_json(path)
    

class WeatherScraper(TimeSeriesScraper):
    def __init__(self):
        super().__init__()
        self.folder = "weather"
        self.raw_url = "https://archive-api.open-meteo.com/v1/archive?latitude={}&longitude={}&start_date={}&end_date={}&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,rain,snowfall,surface_pressure,cloud_cover,wind_speed_10m,wind_direction_10m&timezone=auto"
        
    def to_dataframe(self):
        self.df = pd.DataFrame(self.json['hourly'])
        
    def mass_scrape(self, df, start="2020-01-02", end=None):
        if end is None:
            end = strftime('%Y-%m-%d', gmtime(time()))
        for i in range(df.shape[0]):
            row = df.loc[i]
            self.scrape_and_store(str(row["id"]) + ".csv", row["lat"], row["lng"], start, end)
        
        
if __name__ == "__main__":
    scraper = WeatherScraper()
    df = pd.read_csv("eu.csv")
    scraper.mass_scrape(df, "2024-10-22")
    