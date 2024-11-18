import kivy
from kivy.app import App
from kivy.uix.button import Button
import geopandas as gpd
import folium
import json
import tempfile
import webbrowser

# Create an interactive Folium map with province details on hover
def create_map():
    gdf = gpd.read_file('ds_project/vnmap/vn_shp/vn.shp', encoding = 'utf-8')
    
    # Define tooltip fields (replace with actual fields from your shapefile)
    tooltip_fields = ["id", "name"]

    # Create a Folium map
    m = folium.Map(location=[14.0583, 108.2772], zoom_start=6)

    # Add GeoJSON layer with tooltips for province details
    folium.GeoJson(
        gdf,
        name="Provinces",
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, 
                                      aliases=["ID: ", "Province: "]),
        style_function=lambda x: {"fillColor": "blue", "color": "black", "fillOpacity": 0.5, "weight": 0.5}
    ).add_to(m)

    # Save the map to a temporary file and open it in the default browser
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
        m.save(f.name)
        webbrowser.open("file://" + f.name)

class MapApp(App):
    def build(self):
        return Button(text="Open Interactive Map with Province Details", on_press=lambda x: create_map())

if __name__ == "__main__":
    MapApp().run()
