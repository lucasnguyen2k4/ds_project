import sys
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QCalendarWidget, QLabel, QHBoxLayout, QPushButton, QSpinBox, QMessageBox, QComboBox
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
        self.gdf = gpd.read_file('vn_shp/vn.shp', encoding='utf-8')
        num_provinces = len(self.gdf)
        color_palette = plt.cm.tab20(np.linspace(0, 1, num_provinces))
        colors = ["#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in color_palette]
        self.gdf["color"] = colors

        # Initialize state variables
        self.last_selected_date = None  # To track if the map needs refreshing

        # Initialize UI components
        self.init_ui()

    def init_ui(self):
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
        self.minute_spinbox = QSpinBox()
        self.minute_spinbox.setRange(0, 59)

        self.update_clock_button = QPushButton("Update Clock")
        self.update_clock_button.clicked.connect(self.update_clocks)

        self.browser = QWebEngineView()
        self.update_map(force=True)  # Initial map render

        self.model_combobox1 = QComboBox()
        self.model_combobox1.addItems(["Model A", "Model B", "Model C", "Model D"])
        self.model_combobox2 = QComboBox()
        self.model_combobox2.addItems(["Historical", "Prediction"])
        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.confirm_selection)

        # Layout
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.calendar)
        left_layout.addWidget(self.analog_clock)
        left_layout.addWidget(self.digital_clock_label)

        clock_controls = QHBoxLayout()
        clock_controls.addWidget(QLabel("Hours:"))
        clock_controls.addWidget(self.hour_spinbox)
        clock_controls.addWidget(QLabel("Minutes:"))
        clock_controls.addWidget(self.minute_spinbox)
        clock_controls.addWidget(self.update_clock_button)
        left_layout.addLayout(clock_controls)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.browser)

        left_layout.addWidget(QLabel("Select Model:"))
        left_layout.addWidget(self.model_combobox1)
        left_layout.addWidget(QLabel("Select Mode:"))
        left_layout.addWidget(self.model_combobox2)
        left_layout.addWidget(self.confirm_button)

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

        # Connect signals
        self.calendar.selectionChanged.connect(self.handle_calendar_change)

        # Timer to update digital clock every second
        #self.digital_timer = QTimer(self)
        #self.digital_timer.timeout.connect(self.update_digital_clock)
        #self.digital_timer.start(1000)

    def handle_calendar_change(self):
        selected_date = self.calendar.selectedDate()
        if self.date_requires_update(selected_date):
            self.update_map()
        self.last_selected_date = selected_date

    def date_requires_update(self, selected_date):
        # Check if the date has changed since the last update
        return self.last_selected_date != selected_date

    def update_map(self, force=False):
        if force or self.date_requires_update(self.calendar.selectedDate()):
            self.browser.setHtml(self.create_map())

    def create_map(self):
        m = folium.Map(
            location=[14.0583, 108.2772],
            zoom_start=6,
            max_bounds=True,
            bounds=[[8.179, 102.144], [23.393, 109.463]]
        )

        folium.GeoJson(
            self.gdf,
            name="Provinces",
            style_function=lambda feature: {
                "fillColor": feature["properties"]["color"],
                "color": "black",
                "weight": 0.5,
                "fillOpacity": 0.7,
            },
            tooltip=folium.GeoJsonTooltip(fields=["id", "name"], aliases=["ID:", "Province:"]),
        ).add_to(m)

        return m._repr_html_()

    def update_clocks(self):
        hours = self.hour_spinbox.value()
        minutes = self.minute_spinbox.value()
        time = QTime(hours, minutes, 0)
        if time.isValid():
            self.analog_clock.set_time(time)
            self.digital_clock_label.setText(time.toString("hh:mm"))

    #def update_digital_clock(self):
        #current_time = QTime.currentTime()
        #self.digital_clock_label.setText(current_time.toString("hh:mm"))
    def confirm_selection(self):
        selected_model = self.model_combobox1.currentText()
        selected_mode = self.model_combobox2.currentText()
        QMessageBox.information(self, "Selection Confirmed", f"Model: {selected_model}\nMode: {selected_mode}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MapApp()
    main_window.show()
    sys.exit(app.exec_())
