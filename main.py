from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot, QObject, pyqtSignal, Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QWidget, QTextEdit, QVBoxLayout, QApplication, QGraphicsRectItem, QGraphicsScene, QGraphicsView, QProgressBar
from pyqtgraph.exporters import ImageExporter
#from qtrangeslider import QRangeSlider
#from PyQt5.QtGui import QBrush
#import pyqtgraph as pg
import sys, time, csv, socket
from datetime import datetime
import numpy as np
import nidcpower
from nidcpower import enums as nienums
from mmc103 import *
"""
import pyvisa, serial
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import ipympl
from pathlib import Path
import cv2
"""
import pyqtgraph as pg

def Load_Csv(csv_file_path):
    return np.loadtxt(csv_file_path, delimiter=',')

def Save_Csv(fileName, data):
    csv_file_path = fileName + '.csv'
    np.savetxt(csv_file_path, data, delimiter=',', fmt='%.8f')

def Normalized(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val - min_val == 0:
        return arr - min_val
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr

class EmittingStream(QObject):
    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

class Stats(QMainWindow):

    old_image = pyqtSignal(np.ndarray)
    def __init__(self):
        super().__init__()
        # Load UI
        self.ui = uic.loadUi("ScanGUI.ui", self)
        self.setWindowTitle("ScanGUI")

        # Output Display
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        self.outputTextEdit = self.ui.findChild(QTextEdit, "Console")

        # Parameters
        self.pickup1_enabled = False
        self.pickup2_enabled = False
        self.rect_item = None
        self.ni_stat = False
        self.piezo_state = False
        self.upper = 99
        self.lower = 0
        self.bin_number = 100
        self.raw_data1 = np.rot90(Load_Csv("OIP.csv"), -1)
        self.data2 = np.zeros((1000, 1000))
        
        self.myMMC = None
        self.smu = []
        # Events
        self.ui.ni_init.clicked.connect(self.NI_Init)
        self.ui.set_ni.clicked.connect(self.Set_NI)
        self.ui.piezo_init.clicked.connect(self.Piezo_Init)
        self.ui.go_to_zero.clicked.connect(self.Go_To_Zero)
        self.ui.close_loop.stateChanged.connect(self.Switch_Close_Loop)
        self.ui.set_zero.clicked.connect(self.Set_Zero)
        self.ui.move.clicked.connect(self.Move)
        self.ui.p_mode.toggled.connect(self.Switch_Scan_Mode)
        self.ui.scan.clicked.connect(self.Scan)
        self.ui.select_area.clicked.connect(self.Select_Area)
        self.ui.dynamic_range.setSliderPosition((self.lower, self.upper))
        self.ui.dynamic_range.valueChanged.connect(self.Dynamic_Range)
        self.ui.preview_offset.clicked.connect(self.Preview_Offset)
        self.ui.set_offset.clicked.connect(self.Set_Offset)
        self.ui.img2_to_img1.clicked.connect(self.Img1_to_Img2)
        self.ui.save_csv.clicked.connect(self.Manual_Save_Csv)
        self.ui.save_png.clicked.connect(self.Manual_Save_Png)
        self.ui.stop.setEnabled(False)
        self.ui.stop.clicked.connect(self.Stop)
        self.ui.refresh.clicked.connect(self.Refresh)

        # Form Greyscale Color Map
        colors = [(i, i, i) for i in range(256)]
        self.colormap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 256), color=colors)

        # Figures Initialization
        self.plot1 = self.ui.IMG1.addPlot()
        self.img_item1 = pg.ImageItem()
        self.plot1.addItem(self.img_item1)
        self.img_item1.setLookupTable(self.colormap.getLookupTable())
        self.img_item1.setImage(self.raw_data1)
        self.plot1.setAspectLocked(True)
        self.plot1.setMouseEnabled(x=False, y=False)
        self.plot1.hideAxis('bottom')
        self.plot1.hideAxis('left')

        self.img_item2 = pg.ImageItem()
        self.plot2 = self.ui.IMG2.addPlot()
        self.plot2.addItem(self.img_item2)
        self.img_item2.setLookupTable(self.colormap.getLookupTable())
        self.img_item2.setImage(self.data2)
        self.show()

        self.counts, bin_edges = np.histogram(self.raw_data1, bins=self.bin_number)
        self.bin_widths = np.diff(bin_edges)
        self.bar_width = self.bin_widths.mean()
        self.bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        self.histogram = pg.BarGraphItem(x=self.bin_centers, height=self.counts, width=self.bar_width, brush='r')
        self.dist = self.ui.Distribution.addPlot()
        self.dist.setMouseEnabled(x=False, y=False)
        self.dist.hideAxis('bottom')
        self.dist.hideAxis('left')
        self.dist.addItem(self.histogram)

        self.scene = QGraphicsScene()
        self.img_item1.scene().sigMouseClicked.connect(self.Mouse_Clicked)


    @pyqtSlot(str)
    def normalOutputWritten(self, text):
        cursor = self.outputTextEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.outputTextEdit.setTextCursor(cursor)
        self.outputTextEdit.ensureCursorVisible()

    def Create_Scan_Thread(self):
        self.scan_thread = ScanThread(self.ui)
        self.scan_thread.stats = self
        self.scan_thread.updated_image.connect(self.Handle_Updated_Image)
        self.scan_thread.progress.connect(self.Progress_Update)

    def NI_Init(self):
        try:
            print("NI Initializing ...")
            self.smu = []
            self.smu.append(nidcpower.Session(resource_name='PXI1Slot2', channels='0', reset=True))
            self.smu.append(nidcpower.Session(resource_name='PXI1Slot2', channels='1', reset=True))
            ####################################################################
            # set internal avg filter, moving average
            self.smu[1].samples_to_average = 10
            # Set Output configuration
            self.smu[1].voltage_level = 0  # set output voltage
            self.smu[1].current_limit = 0.01  # set current limit
            self.smu[1].current_limit_range = 0.01
            self.smu[1].current_limit_autorange = True
            self.Ni_Stat(True)
            print("NI Initialized!")
        except:
            self.Ni_Stat(False)
            print("NI Initialization Error")

    def Set_NI(self):
        voltage_lv = self.ui.Voltage_Lv.value()
        cur_lim = self.ui.Cur_Lim.value() # in mA
        try:
            self.smu[1].voltage_level = voltage_lv
            print("Set SMU voltage level to " + str(voltage_lv) + " V.")
            self.smu[1].current_limit_range = cur_lim / 1000 # mA to A
            print("Set SMU current limit range to " + str(cur_lim) + " mA.")
        except:
            self.Ni_Error()
            print("SMU Setting Error")

    def Piezo_Init(self):
        try:
            print("Piezo Stage Initialing ...")
            MMC.initialize(port='COM4', baudrate=38400)
            self.myMMC = MMC()
            print(self.myMMC.getPos(2), self.myMMC.getPos(1))
            self.Piezo_State(True)
            print("Piezo Stage Initialized!")
        except:
            self.Piezo_State(False)
            print("Piezo Stage Initialization Error")

    def Go_To_Zero(self):
        try:
            print("Stage going to Zero ...")
            self.myMMC.moveCnt(2, 50)
            self.myMMC.moveCnt(1, -50)
        except:
            self.Piezo_State(False)
            print("Piezo Stage Error: Cannot Go to Zero.")

    def Set_Zero(self):
        try:
            print("Setting as the new origin point ...")
            self.myMMC.setZero(2)
            self.myMMC.setZero(1)
        except:
            self.Piezo_State(False)
            print("Piezo Stage Error: Cannot Set Zero Point.")

    def Switch_Close_Loop(self):
        CL = self.sender()
        if CL.isChecked():
            try:
                print("Trying to enable close loop mode")
                self.myMMC.enableClosedLoop(2)
                self.myMMC.enableClosedLoop(1)
                self.ui.close_loop.setEnabled(False)
                print("Succeed. Close Loop: ON.")
            except:
                self.ui.close_loop.setChecked(False)
                self.ui.close_loop.setEnabled(False)
                self.Piezo_State(False)
                print("Piezo Stage Error: Cannot Switch to Close Loop")

    def Switch_Scan_Mode(self):
        SC = self.sender()
        if SC.isChecked():
                print("Pulse Mode On")
        else:
                print("Continuous Mode On")

    def Move(self):
        move_x = self.ui.Move_X.value()
        move_y = self.ui.Move_Y.value()
        try:
            print("Trying to move to (" + str(move_x) + ", " + str(move_y) + ")...")
            self.myMMC.move2Pos(2, move_x)
            self.myMMC.move2Pos(1, move_y)
        except:
            self.Piezo_State(False)
            print("Piezo Stage Error: Movement Error.")

    def Select_Area(self):
        self.pickup1_enabled = True
        self.pickup2_enabled = False
        if self.rect_item:
            self.img_item1.scene().removeItem(self.rect_item)
            self.rect_item = None

    def Mouse_Clicked(self, event):
        if event.button() == Qt.LeftButton:
            if self.pickup1_enabled:
                pos = self.img_item1.mapFromScene(event.scenePos())
                i, j = int(pos.x()), int(pos.y())
                if 0 <= i < self.raw_data1.shape[0] and 0 <= j < self.raw_data1.shape[1]:
                    value = self.raw_data1[j, i]
                    print(f"Start Point: ({i}, {j}), value: {value}")
                    self.pickup1_enabled = False
                    self.pickup2_enabled = True
                    self.start_point = 37.5 + (i / self.raw_data1.shape[0] * 675), 37.5 + ((self.raw_data1.shape[1] - j) / self.raw_data1.shape[1] * 675)
            elif self.pickup2_enabled:
                pos = self.img_item1.mapFromScene(event.scenePos())
                i, j = int(pos.x()), int(pos.y())
                if 0 <= i < self.raw_data1.shape[0] and 0 <= j < self.raw_data1.shape[1]:
                    value = self.raw_data1[j, i]
                    print(f"End Point: ({i}, {j}), value: {value}")
                    self.pickup2_enabled = False
                    self.end_point = 37.5 + (i / self.raw_data1.shape[0] * 675), 37.5 + ((self.raw_data1.shape[1] - j) / self.raw_data1.shape[1] * 675)
                    self.draw_rectangle()

    def draw_rectangle(self):
        if self.rect_item:
            self.img_item1.scene().removeItem(self.rect_item)
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        rect_x = min(x1, x2)
        rect_y = min(y1, y2)
        rect_width = abs(x1 - x2)
        rect_height = abs(y1 - y2)
        self.rect_item = QGraphicsRectItem(rect_x, rect_y, rect_width, rect_height)
        self.rect_item.setPen(pg.mkPen('r', width=2))
        self.img_item1.scene().addItem(self.rect_item)

    @pyqtSlot(np.ndarray)
    def Handle_Updated_Image(self, img):
        self.data2 = img
        self.img_item2.setImage(self.data2)

    @pyqtSlot(int)
    def Progress_Update(self, prog):
        print("progress"+str(prog))
        pixel = self.ui.Pixel_X.value() * self.ui.Pixel_Y.value()
        self.ui.progress_bar.setValue(100 * prog/pixel)
        if self.ui.progress_bar.value() == self.ui.progress_bar.maximum():
            self.Go_To_Zero()
            self.Scanning_Lock(True)

    def Update_Img1(self):
        self.img_item1.setImage(self.raw_data1)
        self.dist.removeItem(self.histogram)
        self.counts, bin_edges = np.histogram(self.raw_data1, bins=self.bin_number)
        self.bin_widths = np.diff(bin_edges)
        self.bar_width = self.bin_widths.mean()
        self.bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.histogram = pg.BarGraphItem(x=self.bin_centers, height=self.counts, width=self.bar_width, brush='r')
        self.dist.addItem(self.histogram)

    def Dynamic_Range(self):
        self.lower, self.upper = self.ui.dynamic_range.sliderPosition()
        print(self.lower, self.upper)
        brushes = []
        for i in range(self.bin_number):
            if self.lower <= i <= self.upper:
                brush = pg.mkBrush('r')
            else:
                brush = pg.mkBrush('b')
            brushes.append(brush)
        self.dist.removeItem(self.histogram)
        self.histogram = pg.BarGraphItem(x=self.bin_centers, height=self.counts, width=self.bar_width, brushes=brushes)
        self.dist.addItem(self.histogram)

        self.data1 = np.copy(self.raw_data1)
        low_lim = self.raw_data1.min() + self.lower * (self.raw_data1.max() - self.raw_data1.min())/self.bin_number
        up_lim = self.raw_data1.max() - (self.bin_number - 1 - self.upper) * (self.raw_data1.max() - self.raw_data1.min())/self.bin_number
        self.data1[self.data1 < low_lim] = 0
        self.data1[self.data1 > up_lim] = 1
        mask = (self.data1 > 0) & (self.data1 < 1)
        self.data1[mask] = (self.data1[mask] - low_lim) / (up_lim - low_lim)
        self.img_item1.setImage(self.data1)

    def Offset(self):
        n = self.ui.offset.value()
        odd_even = 0
        if n < 0:
            n = -n
            odd_even = 1
        raw_data1 = np.rot90(self.raw_data1, -1)
        result = np.empty((raw_data1.shape[0], raw_data1.shape[1] - n), dtype=raw_data1.dtype)
        for i in range(raw_data1.shape[0]):
            if i % 2 == odd_even:
                result[i] = raw_data1[i, n:]
            else:
                result[i] = raw_data1[i, :-n]
        return np.rot90(result, 1)

    def Preview_Offset(self):

        self.img_item1.setImage(self.Offset())

    def Set_Offset(self):
        self.raw_data1 = self.Offset()
        self.Update_Img1()

    def Img1_to_Img2(self):
        self.raw_data1 = Normalized(self.data2)
        filename = "AutoSave"+ str(datetime.now().strftime("%Y%m%d_%H%M%S"))
        Save_Csv(filename, self.data2)
        self.Save_Png(filename)
        self.Update_Img1()

    def Save_Png(self, filename):
        exporter = ImageExporter(self.img_item2)
        exporter.export(filename + '.png')

    def Manual_Save_Csv(self):
        filename = self.ui.File_Name.text() + str(datetime.now().strftime("%Y%m%d_%H%M%S"))
        Save_Csv(filename, self.data2)

    def Manual_Save_Png(self):
        filename = self.ui.File_Name.text() + str(datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.Save_Png(filename)

    def Scanning_Lock(self, bool):
        self.ui.stop.setEnabled(not bool)
        self.ui.ni_init.setEnabled(bool)
        self.ui.piezo_init.setEnabled(bool)
        self.ui.p_mode.setEnabled(bool)
        self.ui.scan.setEnabled(bool)
        self.ui.select_area.setEnabled(bool)
        self.ui.P_Rest_Time.setEnabled(bool)
        self.ui.C_Rest_Time.setEnabled(bool)
        self.ui.Range_X.setEnabled(bool)
        self.ui.Pixel_X.setEnabled(bool)
        self.ui.Range_Y.setEnabled(bool)
        self.ui.Pixel_Y.setEnabled(bool)
        if self.ni_stat:
            self.ui.set_ni.setEnabled(bool)
            self.ui.Voltage_Lv.setEnabled(bool)
            self.ui.Cur_Lim.setEnabled(bool)
        if self.piezo_state:
            self.ui.go_to_zero.setEnabled(bool)
            self.ui.move.setEnabled(bool)
            self.ui.set_zero.setEnabled(bool)
            self.ui.Move_X.setEnabled(bool)
            self.ui.Move_Y.setEnabled(bool)

    def Ni_Stat(self, bool):
        self.ni_stat = bool
        self.ui.set_ni.setEnabled(bool)
        self.ui.Voltage_Lv.setEnabled(bool)
        self.ui.Cur_Lim.setEnabled(bool)
        if bool and self.piezo_state:
            self.ui.scan.setEnabled(True)
            self.ui.refresh.setEnabled(True)
            self.ui.stop.setEnabled(False)
        if not bool:
            self.ui.scan.setEnabled(False)
            self.ui.refresh.setEnabled(False)
            if self.ui.stop.isEnabled():
                self.Stop()
                self.ui.stop.setEnabled(False)

    def Piezo_State(self, bool):
        self.piezo_state = bool
        self.ui.go_to_zero.setEnabled(bool)
        self.ui.move.setEnabled(bool)
        self.ui.set_zero.setEnabled(bool)
        self.ui.Move_X.setEnabled(bool)
        self.ui.Move_Y.setEnabled(bool)
        if bool and self.ni_stat:
            self.ui.scan.setEnabled(True)
            self.ui.refresh.setEnabled(True)
            self.ui.stop.setEnabled(False)
        if not bool:
            self.ui.scan.setEnabled(False)
            self.ui.refresh.setEnabled(False)
            if self.ui.stop.isEnabled():
                self.Stop()
                self.ui.stop.setEnabled(False)

    def Scan(self):
        if not self.scan_thread.isRunning():
            print("Start Scanning...")
            self.scan_thread.start()

    def Refresh(self):
        if not self.scan_thread.isRunning():
            print("Start Refreshing...")
            self.scan_thread.Refresh_Scan()

    def Stop(self):
        self.scan_thread.stop()
        print("Scanning Stop.")
        self.Go_To_Zero()
        self.Scanning_Lock(True)

class ScanThread(QThread):
    updated_image = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int)

    def __init__(self, ui):
        QThread.__init__(self)
        self.ui = ui
        self.is_running = False
        self.refresh = False
        self.stats = None

    def run(self):
        self.myMMC = self.stats.myMMC
        self.smu = self.stats.smu
        self.is_running = True
        size_x = self.ui.Range_X.value()  # mm
        size_y = self.ui.Range_Y.value()  # mm
        pixel_x = self.ui.Pixel_X.value()
        pixel_y = self.ui.Pixel_Y.value()
        step_x = size_x / pixel_x
        step_y = size_y / pixel_y
        if self.refresh:
            img = self.stats.data2
            self.refresh = False
        else:
            img = np.zeros((pixel_x, pixel_y))
        self.stats.Scanning_Lock(False)
        cnt = 0
        if self.ui.c_mode.isChecked():
            stepDelay = self.ui.C_Rest_Time.value()  # second
            dwelltime = 11  # in seconds - calculate manually
            freq = dwelltime/pixel_x  # in second
            for j in range(pixel_y):
                try:
                    self.myMMC.moveCnt(2, step_x if j % 2 else -step_x)  # first move
                except:
                    self.stats.Piezo_State(False)
                    print("Piezo Stage Error.")
                start = len(img) - 1 if j % 2 else 0
                end = -1 if j % 2 else len(img)
                step = -1 if j % 2 else 1
                for i in range(start, end, step):
                    if self.is_running == False:
                        break
                    cnt += 1
                    try:
                        img[i, j] = self.smu[1].measure(nienums.MeasurementTypes.CURRENT)
                    except:
                        self.stats.Ni_Stat(False)
                        img[i, j] = np.random.random_sample()
                    try:
                        self.updated_image.emit(img)
                        self.progress.emit(cnt)
                    except:
                        print("Thread Error")
                    time.sleep(freq/5)
                time.sleep(stepDelay)
                if self.is_running == False:
                    break
                try:
                    self.myMMC.moveCnt(1, step_y)
                except:
                    self.stats.Piezo_State(False)
                    print("Piezo Stage Error.")
                time.sleep(stepDelay)

        else:
            stepDelay = self.ui.P_Rest_Time.value()  # second
            try:
                kei.write("smu.source.output = smu.ON")
            except:
                print("Keithley Error.")
            for j in range(pixel_y):
                start = len(img) - 1 if j % 2 else 0
                end = -1 if j % 2 else len(img)
                step = -1 if j % 2 else 1
                for i in range(start, end, step):
                    if self.is_running == False:
                        break
                    cnt += 1
                    try:
                        img[i, j] = self.smu[1].measure(nienums.MeasurementTypes.CURRENT)
                    except:
                        img[i, j] = np.random.random_sample()
                        self.stats.Ni_Stat(False)
                        print("SMU Error.")
                    try:
                        self.progress.emit(cnt)
                        self.updated_image.emit(img)
                    except:
                        print("Thread Error")
                    try:
                        self.myMMC.moveCnt(2, step * step_x)
                    except:
                        self.stats.Piezo_State(False)
                        print("Piezo Stage Error.")
                    time.sleep(stepDelay)
                try:
                    self.myMMC.moveCnt(1, step_y)
                except:
                    self.stats.Piezo_State(False)
                    print("Piezo Stage Error.")
                time.sleep(stepDelay)
                if self.is_running == False:
                    break
            try:
                kei.write("smu.source.output = smu.OFF")
            except:
                print("Keithley Error.")
        self.is_running = False

    def stop(self):
        self.is_running = False

    def Refresh_Scan(self):
        self.refresh = True
        if not self.isRunning():
            self.start()

    def received_image(self, arr):
        self.img = arr

app = QtWidgets.QApplication(sys.argv)
stats = Stats()
stats.Create_Scan_Thread()
stats.show()
app.exec_()
