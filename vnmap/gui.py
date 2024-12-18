import sys
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import branca.colormap as cm
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QMessageBox,
    QCalendarWidget, QLabel, QHBoxLayout, QPushButton, QSpinBox, QComboBox
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QTimer, QTime, Qt, QRectF, QPointF
from PyQt5.QtGui import QPainter, QPolygon, QColor, QPen
from PyQt5.QtCore import QPoint


class AnalogClock(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.time = QTime.currentTime()

        # Timer to update the clock every second
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)  # Update every 1000ms (1 second)

    def set_time(self, time):
        self.time = time
        self.update()

    def update_time(self):
        self.time = self.time.addSecs(1)
        self.update()

    def paintEvent(self, event):
        side = min(self.width(), self.height())
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.scale(side / 200, side / 200)

        # Draw clock face
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(-90, -90, 180, 180)

        # Draw clock border
        painter.setPen(QPen(Qt.black, 2))
        painter.drawEllipse(-90, -90, 180, 180)

        # Draw hour numbers
        painter.setPen(Qt.black)
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        for hour in range(1, 13):
            angle = 30 * hour
            x = 75 * np.sin(np.radians(angle))
            y = -75 * np.cos(np.radians(angle))
            painter.drawText(QRectF(x - 10, y - 10, 20, 20), Qt.AlignCenter, f"{hour}")

        # Draw hour markers
        painter.setPen(QPen(Qt.black, 1))
        for minute in range(60):
            angle = 6 * minute
            x1 = 80 * np.sin(np.radians(angle))
            y1 = -80 * np.cos(np.radians(angle))
            x2 = 88 * np.sin(np.radians(angle))
            y2 = -88 * np.cos(np.radians(angle))
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # Draw hands
        hour_hand = QPolygon([QPoint(-5, 0), QPoint(0, -40), QPoint(5, 0)])
        minute_hand = QPolygon([QPoint(-3, 0), QPoint(0, -60), QPoint(3, 0)])
        second_hand = QPolygon([QPoint(-1, 0), QPoint(0, -70), QPoint(1, 0)])

        current_time = self.time

        # Draw hour hand
        painter.setBrush(Qt.black)
        painter.save()
        painter.rotate(30 * (current_time.hour() + current_time.minute() / 60))
        painter.drawConvexPolygon(hour_hand)
        painter.restore()

        # Draw minute hand
        painter.setBrush(Qt.gray)
        painter.save()
        painter.rotate(6 * (current_time.minute() + current_time.second() / 60))
        painter.drawConvexPolygon(minute_hand)
        painter.restore()

        # Draw second hand
        painter.setBrush(Qt.red)
        painter.save()
        painter.rotate(6 * current_time.second())
        painter.drawConvexPolygon(second_hand)
        painter.restore()


class MapApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Vietnam Map")
        self.setGeometry(100, 100, 1200, 700)

        # Cache data for reuse
        map_df = gpd.read_file("vnmap/vn_shp/vn.shp", encoding='utf-8').drop("id", axis=1)
        city_df = pd.read_csv("data/region/vietnam/cities.csv").loc[:, ["admin_name", "id"]]
        self.base_df = map_df.merge(city_df, left_on="name", right_on="admin_name")
        
        self.weather_db = {i: pd.read_csv("forecast/weather/" + str(i) + ".csv").set_index("time") for i in self.base_df["id"]}
        self.aqi_forest_db = {i: pd.read_csv("forecast/aqi/random_forest/" + str(i) + ".csv").set_index("time") for i in self.base_df["id"]}
        self.aqi_gru_db = {i: pd.read_csv("forecast/aqi/gru/" + str(i) + ".csv").set_index("time") for i in self.base_df["id"]}
        self.aqi_convlstm_db = {i: pd.read_csv("forecast/aqi/conv_lstm/" + str(i) + ".csv").set_index("time") for i in self.base_df["id"]}
        
        # range for colormap
        self.attr_range = {"temperature_2m": (0, 40),
                           "relative_humidity_2m": (0, 100),
                           "dew_point_2m": (0, 30),
                           "precipitation": (0, 10), 
                           "surface_pressure": (900, 1050),
                           "cloud_cover": (0, 100),
                           "wind_speed_10m": (0, 50),
                           "pm2_5": (0, 75),
                           "pm10": (0, 200),
                           "o3": (0, 180),
                           "so2": (0, 350),
                           "no2": (0, 200),
                           "co": (0, 15400)}
        
        # create color map
        color_func = lambda r, g, b: "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
        self.weather_cmap = [color_func(*plt.cm.jet(v)[:3]) for v in range(256)]
        self.aqi_cmap = [color_func(*plt.cm.jet(v)[:3]) for v in range(110, 256)]

        # Initialize UI components
        self.init_ui()

    def init_ui(self):
        """Initialize UI after opening the app."""
        # Components
        self.calendar = QCalendarWidget()
        self.calendar.setFixedSize(600, 250)

        self.analog_clock = AnalogClock()
        self.analog_clock.setFixedSize(300, 300)

        self.digital_clock_label = QLabel()
        self.digital_clock_label.setAlignment(Qt.AlignCenter)
        self.digital_clock_label.setStyleSheet("font-size: 18px;")

        self.hour_spinbox = QSpinBox()
        self.hour_spinbox.setRange(0, 23)
        current_time = QTime.currentTime()
        self.hour_spinbox.setValue(current_time.hour())

        self.browser = QWebEngineView()

        self.model_combobox1 = QComboBox()
        self.model_combobox1.addItems(["Random Forest", "ConvLSTM", "GRU"])
        self.model_combobox2 = QComboBox()
        self.model_combobox2.addItems(["Weather", "AQI Predict"])
        self.model_combobox2.currentIndexChanged.connect(self.handle_combobox)
        self.weather_attr = QComboBox()
        self.weather_attr.addItems(["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
                                    "surface_pressure", "cloud_cover", "wind_speed_10m"])
        self.aqi_attr = QComboBox()
        self.aqi_attr.addItems(["co", "no2", "o3", "so2", "pm2_5", "pm10"])
        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.update_map)

        # Layout
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.calendar)
        left_layout.addWidget(self.analog_clock)
        left_layout.addWidget(self.digital_clock_label)

        clock_controls = QHBoxLayout()
        clock_controls.addWidget(QLabel("Hours:"))
        clock_controls.addWidget(self.hour_spinbox)
        left_layout.addLayout(clock_controls)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.browser)

        self.model_label = QLabel("Select Model:")
        left_layout.addWidget(self.model_label)
        left_layout.addWidget(self.model_combobox1)
        left_layout.addWidget(QLabel("Select Mode:"))
        left_layout.addWidget(self.model_combobox2)
        left_layout.addWidget(self.weather_attr)
        left_layout.addWidget(self.aqi_attr)
        left_layout.addWidget(self.confirm_button)
        
        self.model_label.hide()
        self.model_combobox1.hide()
        self.aqi_attr.hide()

        main_layout = QHBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 2)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
    def handle_combobox(self):
        """Control how comboboxes show and hide based on their values."""
        if self.model_combobox2.currentText() == "Weather":
            self.model_label.hide()
            self.model_combobox1.hide()
            self.aqi_attr.hide()
            self.weather_attr.show()
        else:
            self.model_label.show()
            self.model_combobox1.show()
            self.aqi_attr.show()
            self.weather_attr.hide()
        
    def color_func(self, low, high, cmap):
        return lambda value: cmap[round(min(max((value - low) / (high - low), 0), 1) * (len(cmap) - 1))]
    
    def get_selected_time(self):
        """Get time from hour spinbox and calendar, then make a Datetime string."""
        date = self.calendar.selectedDate()
        day, month, year = date.day(), date.month(), date.year()
        hour = self.hour_spinbox.value()
        return f"{year}-{month:02}-{day:02}T{hour:02}:00"
        
    def set_dataframe(self):
        """Create dataframe for visualization."""
        self.show_df = self.base_df.loc[:]
        selected_time = self.get_selected_time()
        
        if self.model_combobox2.currentText() == "Weather":
            self.attr = self.weather_attr.currentText()
            self.cmap = self.weather_cmap
            db = self.weather_db
        else:
            self.attr = self.aqi_attr.currentText()
            self.cmap = self.aqi_cmap
            if self.model_combobox1.currentText() == "Random Forest":
                db = self.aqi_forest_db
            elif self.model_combobox1.currentText() == "GRU":
                db = self.aqi_gru_db
            else:
                db = self.aqi_convlstm_db
                
        self.low, self.high = self.attr_range[self.attr]
        self.show_df[self.attr] = [max(db[i].loc[selected_time, self.attr], 0) for i in self.base_df["id"]]
        self.show_df["color"] = self.show_df[self.attr].apply(self.color_func(self.low, self.high, self.cmap))

    def update_map(self):
        """Update map when the "Confirm" button is clicked."""
        try:
            self.set_dataframe()
            self.browser.setHtml(self.create_map())
        except:
            message_box = QMessageBox()
            message_box.setWindowTitle("Warning")
            message_box.setText("Chosen date is out of 7-day range from last data scrapping attempt.")
            message_box.exec()
    
    def create_map(self):
        """Draw map to visualize the dataframe after change."""
        m = folium.Map(
            location=[14.0583, 108.2772],
            zoom_start=6,
            max_bounds=True,
            bounds=[[8.179, 102.144], [23.393, 109.463]]
        )
        
        m.add_child(cm.LinearColormap(self.cmap, vmin=self.low, vmax=self.high))

        folium.GeoJson(
            self.show_df,
            name="Provinces",
            style_function=lambda feature: {
                "fillColor": feature["properties"]["color"],
                "color": "black",
                "weight": 0.5,
                "fillOpacity": 0.7,
            },
            tooltip=folium.GeoJsonTooltip(fields=["name", self.attr], aliases=["Province:", self.attr]),
        ).add_to(m)

        return m._repr_html_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MapApp()
    main_window.show()
    sys.exit(app.exec_())
