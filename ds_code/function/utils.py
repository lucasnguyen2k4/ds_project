import pandas as pd
import numpy as np

def group_data(region_folder, src_folder, filename, to_csv=True, src='data'):
    """Group files in a folder to a big file containing infomation of multiple location."""
    keys = []
    city_dfs = []
    
    df = pd.read_csv("data/region/" + region_folder + "/" + "cities.csv")
    for i in range(df.shape[0]):
        row = df.loc[i]
        key = row["city"] + ', ' + row["country"]
        keys.append(key)
        city_df = pd.read_csv(src + '/' + src_folder + "/" + str(row["id"]) + ".csv").set_index("time")
        city_dfs.append(city_df)
        
    region_df = pd.concat(city_dfs, axis=1, keys=keys)
    if to_csv:
        region_df.to_csv("data/region/" + region_folder + "/" + filename)
    else:
        return region_df
    
    
def group_weather_data(region_folder, filename="weather.csv"):
    group_data(region_folder, "weather", filename)
    
    
def group_aqi_data(region_folder, filename="air_quality.csv"):
    group_data(region_folder, "air_quality", filename)
    
    
def read_group_data(path_from_region):
    path = "data/region/" + path_from_region
    return pd.read_csv(path, index_col=0, header=[0, 1])


def sliding_window(weather_df, air_df, window_size=4, target_size="same"):           # target_size is either "one" or "same"
    """
    Create windows for data preprocessing step, with n hours of weather data
    and corresponding 1 hour of AQI data (target_size="one") or n hours of AQI
    data (target_size="same").
    """
    X = []
    y = []
    w_df = weather_df[:]
    a_df = air_df[:]
    w_df["time"] = pd.to_datetime(w_df["time"])
    a_df["time"] = pd.to_datetime(a_df["time"])
    w_df.set_index("time", inplace=True)
    a_df.set_index("time", inplace=True)
    weather_time_idx = w_df.index
    
    for i in range(window_size - 1, len(weather_df)):
        w_start, w_end = weather_time_idx[i - window_size + 1], weather_time_idx[i]
        if (w_end - w_start).seconds == 3600 * (window_size - 1):
            try:
                if target_size == "one":
                    a_value = a_df.loc[w_end].values
                elif target_size == "same":
                    a_value = a_df.loc[w_start: w_end].values
                    if len(a_value) != window_size:
                        continue 
                    
                w_value = w_df.loc[w_start: w_end].values
                X.append(w_value)
                y.append(a_value)
            except:
                continue
            
    return np.array(X), np.array(y)

def predict_window(weather_df, window_size=4):
    """
    Create windows for data preprocessing step of large scale prediction, 
    using whole weather data table of a province.
    """
    X = []
    valid = []
    w_df = weather_df[:]
    w_df["time"] = pd.to_datetime(w_df["time"])
    w_df.set_index("time", inplace=True)
    weather_time_idx = w_df.index
    
    for i in range(window_size - 1, len(weather_df)):
        w_start, w_end = weather_time_idx[i - window_size + 1], weather_time_idx[i]
        if (w_end - w_start).seconds == 3600 * (window_size - 1):
            w_value = w_df.loc[w_start: w_end].values
            X.append(w_value)
            valid.append(i)
            
    return weather_time_idx[valid], np.array(X)

    
if __name__ == "__main__":
    pass