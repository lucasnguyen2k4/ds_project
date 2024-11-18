import pandas as pd
from function.scraper import AQIScraper, WeatherScraper
from function.utils import *

# create dataframe for the region
df = pd.read_csv("data/countries/worldcities.csv")
vn_df = df[((df["capital"] == "admin") | (df["capital"] == "primary")) & (df["country"] == "Vietnam")]
vn_df.to_csv("data/region/vietnam/cities.csv", index=False)
df = pd.read_csv("data/region/vietnam/cities.csv")

# API for scraping
weather_scraper = WeatherScraper()
aqi_scraper = AQIScraper()

# scraping
weather_scraper.mass_scrape(df)
aqi_scraper.mass_scrape(df)

# concanate multiple city files --> regional files
group_weather_data("vietnam")
group_aqi_data("vietnam")