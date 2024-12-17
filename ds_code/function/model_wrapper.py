import joblib
import pandas as pd
import numpy as np
from torch import load
from utils import predict_window
from models import *

class ModelWrapper:
    """Wrapper for convenient large scale prediction with models."""
    def __init__(self, model_dir, output_dir):
        self.model = self.load_model(model_dir)
        self.output_dir = output_dir
        self.input_dir = "forecast/weather"
        
    def load_model(self, model_dir):
        pass
        
    def input_process(self, X, init_data):
        pass
    
    def get_dataframe(self, i):
        pass
        
    def forecast(self, extra_dir):
        """Method for predicting and storing the result for further visualization."""
        init_df = pd.read_csv(extra_dir).set_index("id")
        for i in init_df.index:
            time_idx, X = predict_window(self.get_dataframe(i))
            init_data = init_df.loc[i]
            inp = self.input_process(X, init_data)
            output = self.model.predict(inp)
            forecast_df = pd.DataFrame(output, index=time_idx.strftime("%Y-%m-%dT%H:%M"), 
                                       columns=["co", "no2", "o3", "so2", "pm2_5", "pm10"]).apply(lambda x: round(x, 2))
            forecast_df.to_csv(f"{self.output_dir}/{i}.csv")
            
class RandomForestPredictor(ModelWrapper):
    def __init__(self):
        super().__init__("models/random_forest.pkl", "forecast/aqi/random_forest")
        
    def load_model(self, model_dir):
        return joblib.load(model_dir)    
    
    def get_dataframe(self, i):
        return pd.read_csv(f"{self.input_dir}/{i}.csv")
        
    def input_process(self, X, init_data):
        m = X.shape[0]
        lat = np.full((m, 1), init_data[0])
        lng = np.full((m, 1), init_data[1])
        population = np.full((m, 1), init_data[2])
        init_data = np.hstack((lat, lng, population))
        return np.hstack((X.reshape(m, -1), init_data))
    
class GRUPredictor(ModelWrapper):
    def __init__(self):
        super().__init__("models/gru.pth", "forecast/aqi/gru")
        
    def load_model(self, model_dir):
        return load(model_dir, map_location="cpu")
    
    def get_dataframe(self, i):
        weather_df = pd.read_csv(f"{self.input_dir}/{i}.csv")
        weather_df["wind_x_component"] = np.cos(weather_df["wind_direction_10m"] / (180 / np.pi))
        weather_df["wind_y_component"] = np.sin(weather_df["wind_direction_10m"] / (180 / np.pi))
        weather_df.drop("wind_direction_10m", axis=1, inplace=True)
        return weather_df
    
    def input_process(self, X, init_data):
        m = X.shape[0]
        lat = np.full((m, 1), init_data[0])
        lng = np.full((m, 1), init_data[1])
        population = np.full((m, 1), init_data[2])
        init_data = np.hstack((lat, lng, population))
        return X, init_data
    
if __name__ == "__main__":
    predictor = GRUPredictor()
    predictor.forecast(extra_dir="data/region/vietnam/extra_info.csv") 
    #print(pd.read_csv("forecast/weather/1704000203.csv"))
        
        