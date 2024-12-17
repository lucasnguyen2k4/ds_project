import os
import requests
import json
import pandas as pd
from time import time, sleep
from datetime import datetime

class Scraper:
    def __init__(self):
        self.raw_url = None
        self.folder = None
        self.json = None
        self.df = None
        
    def get_json(self, *args):
        url = self.raw_url.format(*args)
        response = requests.get(url)
        print(url)
        
        if response.status_code == 429:
            print("API request limit exceeded, retry in 30 seconds")
            sleep(30)
            self.get_json(*args)
        elif response.status_code == 200:
            print("Request OK")
            self.json = response.json()
            return True
        else:
            print(f'Error: Request failed with status code {response.status_code}')
            return False
        
    def to_dataframe(self):
        pass
        
    def store(self, filename):
        #self.df.dropna(thresh=3, inplace=True)
        path = self.folder + "\\" + filename
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
        self.folder = "data/countries"
        
    def to_dataframe(self):
        path = "data" + "\\" + self.folder + "\\" + 'countries.json'
        with open(path, 'w') as json_file:
            json.dump(self.json["geonames"], json_file, indent=4)  
        self.df = pd.read_json(path)
    

class WeatherScraper(TimeSeriesScraper):
    def __init__(self):
        super().__init__()
        self.folder = "data/weather"
        self.raw_url = "https://archive-api.open-meteo.com/v1/archive?latitude={}&longitude={}&start_date={}&end_date={}&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,surface_pressure,cloud_cover,wind_speed_10m,wind_direction_10m&timezone=auto"
        
    def to_dataframe(self):
        self.df = pd.DataFrame(self.json['hourly'])
        
    def mass_scrape(self, df, start="2020-11-28", end="now", overwrite=True):
        if end == "now":
            end = datetime.fromtimestamp(time()).strftime("%Y-%m-%d")
            
        for i in range(df.shape[0]):
            row = df.loc[i]
            if overwrite or not os.path.exists("data" + "\\" + self.folder + "\\" + str(row["id"]) + ".csv"):
                self.scrape_and_store(str(row["id"]) + ".csv", row["lat"], row["lng"], start, end)
                
    def forecast_scrape(self, df):
        self.folder = "forecast/weather"
        self.raw_url = "https://api.open-meteo.com/v1/forecast?latitude={}&longitude={}&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,surface_pressure,cloud_cover,wind_speed_10m,wind_direction_10m&timezone=auto&past_days=1"
        
        for i in range(df.shape[0]):
            row = df.loc[i]
            self.scrape_and_store(str(row["id"]) + ".csv", row["lat"], row["lng"])    

class AQIScraper(TimeSeriesScraper):
    def __init__(self):
        super().__init__()
        self.folder = "data/air_quality"
        self.raw_url = "http://api.openweathermap.org/data/2.5/air_pollution/history?lat={}&lon={}&start={}&end={}&appid=b34c8120213e3f26c596cfb41b21cb86"
        
    def to_dataframe(self):
        json = str({datetime.fromtimestamp(obj["dt"]).strftime("%Y-%m-%dT%H:%M:%S"): obj["components"] | obj["main"] for obj in self.json["list"]}).replace('\'', '"')
        self.df = pd.read_json(json, orient="index").drop(["no", "nh3"], axis=1)
        self.df.index.name = "time"
        self.df.reset_index(inplace=True)
        
    def mass_scrape(self, df, start="2020-11-28", end="now", overwrite=True):
        start_stamp = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
        if end == "now":
            end_stamp = int(time())
        else:
            end_stamp = int(datetime.strptime(end, "%Y-%m-%d").timestamp())

        for i in range(df.shape[0]):
            row = df.loc[i]
            if overwrite or not os.path.exists("data" + "\\" + self.folder + "\\" + str(row["id"]) + ".csv"):
                self.scrape_and_store(str(row["id"]) + ".csv", row["lat"], row["lng"], start_stamp, end_stamp)
        
    
if __name__ == "__main__":
    """
    df = pd.read_csv("data/region/vietnam/cities.csv")
    scraper = WeatherScraper()
    scraper.forecast_scrape(df)
    """