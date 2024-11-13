import sys
import geopandas as gpd
import folium
import json
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView

# Function to create the interactive map
def create_map():
    # Load the shapefile with geopandas
    gdf = gpd.read_file('vn_shp/vn.shp', encoding = 'utf-8')

    # Generate a unique color for each province using a color palette
    num_provinces = len(gdf)
    color_palette = plt.cm.tab20(np.linspace(0, 1, num_provinces))
    colors = ["#%02x%02x%02x" % (int(r*255), int(g*255), int(b*255)) for r, g, b, _ in color_palette]

    # Map each province's index to a unique color
    gdf["color"] = colors

    # Create Folium map
    m = folium.Map(location=[14.0583, 108.2772], zoom_start=6)

    # Define the GeoJSON layer with a style function that assigns each province its unique color
    folium.GeoJson(
        gdf,
        name="Provinces",
        style_function=lambda feature: {
            "fillColor": feature["properties"]["color"],
            "color": "black",
            "weight": 0.5,
            "fillOpacity": 0.7,
        },
        tooltip=folium.GeoJsonTooltip(fields=["id", "name"], aliases=["ID: ","Province: "]),  # Replace with your actual field
    ).add_to(m)

    # Render map as HTML
    return m._repr_html_()

# PyQt5 GUI to display the interactive map
class MapApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Map with Unique Province Colors in PyQt5")
        self.setGeometry(100, 100, 800, 600)

        # Generate and embed the map in QWebEngineView
        self.browser = QWebEngineView()
        self.browser.setHtml(create_map())
        
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.browser)
        self.setCentralWidget(central_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MapApp()
    main_window.show()
    sys.exit(app.exec_())
