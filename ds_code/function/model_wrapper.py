import joblib
import pandas as pd
import numpy as np
from torch import load
from utils import predict_window, group_data
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
    
class ConvLSTMPredictor(ModelWrapper):
    def __init__(self):
        super().__init__("models/conv_lstm.pth", "forecast/aqi/conv_lstm")

    def load_model(self, model_dir):
        conv_lstm = ConvLSTMTimeSeries(
            input_dim = 63,
            hidden_dim = [256],
            input_width = 9,
            output_width = 6
        )
        conv_lstm.load_state_dict(torch.load(model_dir, map_location="cpu", weights_only=True))
        return conv_lstm

    def get_dataframe(self):
        weather_df = group_data('vietnam', 'weather' , '', to_csv=False, src = 'forecast')
        self.time_index = weather_df.index
        weather_df = weather_df.stack(level=0).reset_index()
        weather_df.rename(columns={'level_1': 'location'}, inplace=True)
        weather_df[['province', 'country']] = weather_df['location'].str.split(',', expand=True)
        weather_df = weather_df.drop(columns=['location'])

        cities_df = pd.read_csv("data/region/vietnam/cities.csv")
        weather_df = weather_df.merge(cities_df[['city', 'admin_name']], right_on='city', left_on='province', how='left')
        weather_df.drop(columns=['country', 'city', 'province'], inplace=True)
        weather_df.rename(columns={'admin_name': 'province'}, inplace=True)
        weather_df['time'] = weather_df['time'].astype('datetime64[s]')
        weather_df = weather_df.set_index(['time', 'province']).sort_index()

        weather_df["wind_x_component"] = np.cos(weather_df["wind_direction_10m"] / (180 / np.pi))
        weather_df["wind_y_component"] = np.sin(weather_df["wind_direction_10m"] / (180 / np.pi))
        weather_df.drop(columns='wind_direction_10m', inplace=True)
        weather_df = weather_df.reset_index().sort_values(by=['province', 'time'])
        self.province = weather_df['province']
        return weather_df

    def forecast(self):
        custom_scaler = joblib.load('models/weather_scaler.pickle'), joblib.load('models/air_scaler.pickle')
        weather_df = self.get_dataframe()
        predict_dataset =  TimeSeries3DDataset(None, weather_df.drop(columns=['province', 'time']), 63, 3, custom_scaler=custom_scaler, predict=True)
        predict_dataloader = DataLoader(predict_dataset, batch_size=1, shuffle=False, num_workers = 0, pin_memory=True)

        with torch.no_grad():
            outputs = []
            for X in predict_dataloader:
                output = self.model.predict(X, numpy_output=False).squeeze()
                output = output.view(63, 1, 6)
                outputs.append(output)

            stacked_outputs = torch.cat(outputs, dim=1)
            original_outputs = stacked_outputs.view(-1, 6)
            original_outputs = predict_dataset.target_scaler.inverse_transform(original_outputs)
        length, width = original_outputs.shape
        ids = pd.read_csv('data/region/vietnam/cities.csv').set_index("admin_name").loc[self.province.sort_index()[:63]]['id']
        for i in range(63):
            forecast_df = pd.DataFrame(original_outputs[i*length//63:(i+1)*length//63], index=self.time_index, 
                                       columns=["co", "no2", "o3", "so2", "pm2_5", "pm10"]).apply(lambda x: round(x, 2))
            forecast_df.to_csv(f"{self.output_dir}/{ids[i]}.csv")
            

if __name__ == "__main__":
    pass
    #predictor = GRUPredictor()
    #predictor.forecast(extra_dir="data/region/vietnam/extra_info.csv") 
    ConvLSTMPredictor().forecast()
    #print(pd.read_csv("forecast/weather/1704000203.csv"))
        
        