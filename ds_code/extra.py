# generate "extra_info.csv" file in data folder
import pandas as pd

city_df = pd.read_csv("data/region/vietnam/cities.csv")

city_data = city_df.loc[:, ["id", "lat", "lng", "population"]]
city_data["population"].fillna(city_data["population"].mean(), inplace=True)
city_data.set_index("id", inplace=True)

city_data.to_csv("data/region/vietnam/extra_info.csv")