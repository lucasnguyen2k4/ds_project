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

# test run, for full scraping --> only keep 'df' argument
weather_scraper.mass_scrape(df, "2024-10-27")
aqi_scraper.mass_scrape(df, "2024-10-27")

# concanate multiple city files --> regional files
regional_weather_build("vietnam")
regional_aqi_build("vietnam")