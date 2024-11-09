import pandas as pd

def regional_data_build(region_folder, src_folder, filename):
    keys = []
    city_dfs = []
    
    df = pd.read_csv("data/region/" + region_folder + "/" + "cities.csv")
    for i in range(df.shape[0]):
        row = df.loc[i]
        key = row["city"] + ', ' + row["country"]
        keys.append(key)
        city_df = pd.read_csv("data/" + src_folder + "/" + str(row["id"]) + ".csv").set_index("time")
        city_dfs.append(city_df)
        
    region_df = pd.concat(city_dfs, axis=1, keys=keys).dropna()
    region_df.to_csv("data/region/" + region_folder + "/" + filename)
    
    
def regional_weather_build(region_folder, filename="weather.csv"):
    regional_data_build(region_folder, "weather", filename)
    
    
def regional_aqi_build(region_folder, filename="air_quality.csv"):
    regional_data_build(region_folder, "air_quality", filename)
    
    
def read_regional_data(path_from_region):
    path = "data/region/" + path_from_region
    return pd.read_csv(path, index_col=0, header=[0, 1])

    
if __name__ == "__main__":
    print(read_regional_data("vietnam/weather.csv"))