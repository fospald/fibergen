#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, division
import builtins

import fibergen
import sys, os, re
import webbrowser
import base64
import copy
import pydoc
import traceback
import codecs
import collections
import argparse
import tempfile
import subprocess
import xml.etree.ElementTree as ET
import keyword
import textwrap
import signal
from weakref import WeakKeyDictionary
from html import escape as html_escape

try:
	import numpy as np
	import scipy.misc
	from PyQt5 import QtCore, QtGui, QtWidgets

	try:
		from PyQt5 import QtWebKitWidgets
	except:
		from PyQt5 import QtWebEngineWidgets as QtWebKitWidgets
		QtWebKitWidgets.QWebView = QtWebKitWidgets.QWebEngineView
		QtWebKitWidgets.QWebPage = QtWebKitWidgets.QWebEnginePage

	import matplotlib
	from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
	try:
		from matplotlib.backends.backend_qt5agg import NavigationToolbar2QTAgg as NavigationToolbar
	except:
		from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

	from matplotlib.figure import Figure
	from matplotlib.backend_bases import cursors
	import matplotlib.pyplot as plt

	from matplotlib import rcParams
	import matplotlib.ticker as mtick
	import matplotlib.cm as mcmap

except BaseException as e:
	print(str(e))
	print("Make sure you have the scipy, numpy, matplotlib, pyqt5 and pyqt5-webengine packages for Python%d installed!" % sys.version_info[0])
	sys.exit(1)


class PreferencesWidget(QtWidgets.QDialog):

	def __init__(self, parent=None):
		super(PreferencesWidget, self).__init__(parent)

		app = QtWidgets.QApplication.instance()

		self.setWindowTitle("Preferences")
		self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
		
		grid = QtWidgets.QGridLayout()

		self.fontCombo = QtWidgets.QFontComboBox()
		self.fontCombo.setCurrentText(app.window.textEdit.font().family())
		row = grid.rowCount()
		grid.addWidget(QtWidgets.QLabel("Font:"), row, 0)
		grid.addWidget(self.fontCombo, row, 1)
		
		self.fontSize = QtWidgets.QSpinBox()
		self.fontSize.setMinimum(1)
		self.fontSize.setMaximum(100)
		self.fontSize.setValue(app.window.textEdit.font().pointSize())
		row = grid.rowCount()
		grid.addWidget(QtWidgets.QLabel("Font size:"), row, 0)
		grid.addWidget(self.fontSize, row, 1)

		self.tabSize = QtWidgets.QSpinBox()
		self.tabSize.setMinimum(1)
		self.tabSize.setMaximum(1000)
		self.tabSize.setValue(app.window.textEdit.tabStopWidth())
		row = grid.rowCount()
		grid.addWidget(QtWidgets.QLabel("Tab width:"), row, 0)
		grid.addWidget(self.tabSize, row, 1)

		hline = QtWidgets.QFrame()
		hline.setFrameShape(QtWidgets.QFrame.HLine)
		hline.setFrameShadow(QtWidgets.QFrame.Sunken)
		row = grid.rowCount()
		grid.addWidget(hline, row, 0, row, 2)

		hbox = QtWidgets.QHBoxLayout()
		okButton = QtWidgets.QPushButton("&Save")
		okButton.clicked.connect(self.save)
		cancelButton = QtWidgets.QPushButton("&Cancel")
		cancelButton.clicked.connect(self.close)

		hbox.addStretch(1)
		hbox.addWidget(cancelButton)
		hbox.addWidget(okButton)
		row = grid.rowCount()
		grid.addLayout(hbox, row, 0, row, 2)

		self.setLayout(grid)

	def save(self):

		app = QtWidgets.QApplication.instance()

		font = self.fontCombo.currentFont()
		font.setPointSize(self.fontSize.value())

		if font.family() != app.window.textEdit.font().family():
			app.settings.setValue("fontFamily", font.family())
		if font.pointSize() != app.window.textEdit.font().pointSize():
			app.settings.setValue("fontPointSize", font.pointSize())

		app.window.textEdit.setFont(font)

		tabSize = self.tabSize.value()
		if tabSize != app.window.textEdit.tabStopWidth():
			app.window.textEdit.setTabStopWidth(tabSize)
			app.settings.setValue("tabStopWidth", tabSize)

		self.close()


class WriteVTKWidget(QtWidgets.QDialog):

	def __init__(self, filename, rve_dims, field_groups, parent=None):
		super(WriteVTKWidget, self).__init__(parent)

		self.setWindowTitle("Write VTK")
		self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
		
		self.rve_dims = rve_dims
		self.field_groups = field_groups
		self.filename = filename

		grid = QtWidgets.QGridLayout()
		vbox = QtWidgets.QVBoxLayout()
		grid.addLayout(vbox, 0, 0)

		vbox.addWidget(QtWidgets.QLabel("Fields to export:"))

		for j, field_group in enumerate(field_groups):
			gbox = QtWidgets.QHBoxLayout()
			for i, field in enumerate(field_group):
				field.check = QtWidgets.QCheckBox(field.label)
				field.check.setToolTip(field.description)
				field.check.setChecked(True)
				gbox.addWidget(field.check)
			gbox.addStretch(1)
			vbox.addLayout(gbox)

		hline = QtWidgets.QFrame()
		hline.setFrameShape(QtWidgets.QFrame.HLine)
		hline.setFrameShadow(QtWidgets.QFrame.Sunken)
		vbox.addWidget(hline)

		gbox = QtWidgets.QVBoxLayout()
		self.runParaviewCheck = QtWidgets.QCheckBox("Open with ParaView after save")
		gbox.addWidget(self.runParaviewCheck)
		vbox.addLayout(gbox)

		hbox = QtWidgets.QHBoxLayout()
		okButton = QtWidgets.QPushButton("&Save")
		okButton.clicked.connect(self.writeVTK)
		cancelButton = QtWidgets.QPushButton("&Cancel")
		cancelButton.clicked.connect(self.close)

		hbox.addStretch(1)
		hbox.addWidget(cancelButton)
		hbox.addWidget(okButton)
		grid.addLayout(hbox, 1, 0)

		self.setLayout(grid)

	def writeVTK(self):

		# https://bitbucket.org/pauloh/pyevtk

		app = QtWidgets.QApplication.instance()
		binary = True
		loadstep = 0
		dtype = "float"
		
		with open(self.filename, "wb+") as f:

			def write(s):
				f.write(s.encode("ascii"))

			write("# vtk DataFile Version 2.0\n" + app.applicationName() + "\n")

			if binary:
				write("BINARY\n")
			else:
				write("ASCII\n")

			dummy, nx, ny, nz = self.field_groups[0][0].data[loadstep].shape
			x0, y0, z0, dx, dy, dz = self.rve_dims
			sx, sy, sz = [dx/nx, dy/ny, dz/nz]
			
			write("DATASET STRUCTURED_POINTS\n")
			write("DIMENSIONS " + str(nx+1) + " " + str(ny+1) + " " + str(nz+1) + "\n")
			write("ORIGIN " + str(x0) + " " + str(y0) + " " + str(z0) + "\n")
			write("SPACING " + str(sx) + " " + str(sy) + " " + str(sz) + "\n")

			write("CELL_DATA " + str(nx*ny*nz) + "\n")

			for j, field_group in enumerate(self.field_groups):

				write_componentwise = False
				field_names = []
				has_magnitude = False
				for i, field in enumerate(field_group):
					write_componentwise = write_componentwise or not field.check.isChecked()
					field_names.append(field.name)
					has_magnitude = has_magnitude or field.name.startswith("magnitude_")

				for i, field in enumerate(field_group):

					if not field.check.isChecked():
						continue

					if sys.byteorder == "little":
						data = field.data[loadstep].byteswap()
					else:
						data = field.data[loadstep]

					write("\n")

					ncomp = len(field_group)
					if has_magnitude:
						ncomp -= 1

					if field.name.startswith("magnitude_"):
						ncomp = 1

					if ncomp == 1 or field.name in ["phi"] or write_componentwise:
						write("SCALARS")
						ncomp = 1
					elif ncomp == 3 and field.name in ["n"]:
						write("NORMALS")
					elif ncomp == 3:
						write("VECTORS")
					elif ncomp == 6:
						ncomp = 9
						data = data[[0, 5, 4, 5, 1, 3, 4, 3, 2]]
						write("TENSORS")
						# TODO: check if TENSOR6 is supported by paraview version xxx
						#data = data[[0, 5, 4, 1, 3, 2]]
						#write("TENSORS6")
					elif ncomp == 9:
						data = data[[0, 5, 4, 8, 1, 3, 7, 6, 2]]
						write("TENSORS")
					else:
						print(data.shape, loadstep, field.name, ncomp)
						raise "problem"

					npdtype = field_group[0].data[loadstep].dtype

					if npdtype == np.float32:
						dtype = "float"
					elif npdtype == np.float64:
						dtype = "double"
					elif npdtype == np.int32:
						dtype = "int"
					elif npdtype == np.int64:
						dtype = "long"
					else:
						print(npdtype)
						raise "problem"

					if field.name in ["phi"]:
						label = field.label if not field.label in field_names else field.key
					else:
						label = field.key if write_componentwise else field.name

					write(" " + label + " " + dtype + "\n")

					if ncomp == 1:
						write("LOOKUP_TABLE default\n")

					#print(label, data.shape, ncomp, i, dtype)
					data = data[(i*ncomp):(i*ncomp + ncomp)].tobytes(order='F')

					f.write(data)

					del data

					if ncomp > 1:
						break

		self.close()

		if self.runParaviewCheck.isChecked():
			subprocess.Popen(["paraview", self.filename], cwd=os.path.dirname(self.filename))


class FlowLayout(QtWidgets.QLayout):
	def __init__(self, parent=None, margin=0):
		super(FlowLayout, self).__init__(parent)

		if parent is not None:
			self.setContentsMargins(margin, margin, margin, margin)

		self.itemList = []

	def __del__(self):
		item = self.takeAt(0)
		while item:
			item = self.takeAt(0)

	def addLayout(self, item):
		self.addItem(item)

	def addItem(self, item):
		self.itemList.append(item)

	def addStretch(self, stretch=0):
		s = QtWidgets.QSpacerItem(0, 0)
		self.addItem(s)

	def addSpacing(self, spacing):
		s = QtWidgets.QSpacerItem(spacing, 1)
		self.addItem(s)

	def count(self):
		return len(self.itemList)

	def itemAt(self, index):
		if index >= 0 and index < len(self.itemList):
			return self.itemList[index]

		return None

	def takeAt(self, index):
		if index >= 0 and index < len(self.itemList):
			return self.itemList.pop(index)

		return None

	def expandingDirections(self):
		return QtCore.Qt.Orientations(QtCore.Qt.Orientation(0))

	def hasHeightForWidth(self):
		return True

	def heightForWidth(self, width):
		height = self.doLayout(QtCore.QRect(0, 0, width, 0), True)
		return height

	def setGeometry(self, rect):
		super(FlowLayout, self).setGeometry(rect)
		self.doLayout(rect, False)

	def sizeHint(self):
		return self.minimumSize()

	def minimumSize(self):
		size = QtCore.QSize()

		for item in self.itemList:
			size = size.expandedTo(item.minimumSize())

		margin, _, _, _ = self.getContentsMargins()

		size += QtCore.QSize(2 * margin, 2 * margin)
		return size

	def doLayout(self, rect, testOnly):
		x = rect.x()
		y = rect.y()
		lineHeight = 0

		app = QtWidgets.QApplication.instance()
		spaceY = self.spacing() # app.style().layoutSpacing(QtWidgets.QSizePolicy.PushButton, QtWidgets.QSizePolicy.PushButton, QtCore.Qt.Vertical)
		spaceX = 0 # 2*app.style().layoutSpacing(QtWidgets.QSizePolicy.PushButton, QtWidgets.QSizePolicy.PushButton, QtCore.Qt.Horizontal)

		stretch = False

		for item in self.itemList:

			dx = item.sizeHint().width()

			if isinstance(item, QtWidgets.QSpacerItem):
				stretch = (dx == 0)
				x += dx
				continue

			if stretch:
				x = max(x, rect.right() - dx)
			
			nextX = x + dx

			if x > rect.x() and nextX > rect.right():
				x = max(rect.x(), rect.right() - dx) if stretch else rect.x()
				y = y + lineHeight + spaceY
				nextX = x + dx

			if not testOnly:
				item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), item.sizeHint()))

			x = nextX + spaceX
			lineHeight = max(lineHeight, item.sizeHint().height())
			stretch = False

		return y + lineHeight - rect.y()
	

class MyWebPage(QtWebKitWidgets.QWebPage):

	linkClicked = QtCore.pyqtSignal('QUrl')

	def __init__(self, parent = None):
		super(QtWebKitWidgets.QWebPage, self).__init__(parent)

		try:
			self.setLinkDelegationPolicy(QtWebKitWidgets.QWebPage.DelegateAllLinks)
			self.setHtml = self.setHtmlFrame
			self.acceptNavigationRequest = self.acceptNavigationRequestWebkit
			self.setUrl = self.setUrlFrame
		except:
			pass

	def acceptNavigationRequest(self, url, navigationType, isMainFrame):
		if navigationType == QtWebKitWidgets.QWebPage.NavigationTypeLinkClicked:
			self.linkClicked.emit(url)
			return False
		return QtWebKitWidgets.QWebPage.acceptNavigationRequest(self, url, navigationType, isMainFrame)

	def acceptNavigationRequestWebkit(self, frame, request, navigationType):
		if navigationType == QtWebKitWidgets.QWebPage.NavigationTypeLinkClicked:
			url = request.url()
			self.linkClicked.emit(url)
			return False
		return QtWebKitWidgets.QWebPage.acceptNavigationRequest(self, frame, request, navigationType)

	def setHtmlFrame(self, html):
		self.currentFrame().setHtml(html)

	def setUrlFrame(self, url):
		self.currentFrame().setUrl(url)


def defaultCSS(tags=True):

	app = QtWidgets.QApplication.instance()
	pal = app.palette()
	font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.GeneralFont)

	html = """
body, table {
	font-size: """ + str(font.pointSize()) + """pt;
}
body {
	font-family: \"""" + font.family() + """\";
	background-color: """ + pal.base().color().name() + """;
	color: """ + pal.text().color().name() + """;
}
a {
	color: """ + pal.link().color().name() + """;
	text-decoration: none;
}
a:hover {
	color: """ + pal.link().color().lighter().name() + """;
}
table {
	border-collapse: collapse;
	background-color: """ + pal.window().color().name() + """;
}
.plot {
	border: 1px solid """ + pal.shadow().color().name() + """;
	background-color: """ + pal.window().color().name() + """;
}
th, td, .help {
	border: 1px solid """ + pal.shadow().color().name() + """;
	padding: 0.5em;
	text-align: left;
}
.help {
	background-color: """ + pal.toolTipBase().color().name() + """;
	color: """ + pal.toolTipText().color().name() + """;
	display: inline-block;
}
.help:first-letter {
	text-transform: uppercase;
}
p {
	margin-bottom: 0.5em;
	margin-top: 0.5em;
}
h1, h2, h3 {
	margin: 0;
	margin-bottom: 0.5em;
	padding: 0;
	white-space: nowrap;
}
h1 {
	font-size: """ + str(int(2.0*font.pointSize())) + """pt;
}
h2 {
	font-size: """ + str(int(1.5*font.pointSize())) + """pt;
	margin-top: 1.0em;
}
h3 {
	font-size: """ + str(int(font.pointSize())) + """pt;
	margin-top: 1.0em;
}
"""
	if tags:
		html = "<style>" + html + "</style>"

	return html


class PlotField(object):
	pass

class PlotWidget(QtWidgets.QWidget):

	def __init__(self, rve_dims, field_groups, extra_fields, xml, xml_root, resultText, other = None, parent = None):

		app = QtWidgets.QApplication.instance()
		pal = app.palette()

		self.field_groups = field_groups
		self.rve_dims = rve_dims
		self.view_extra_fields = extra_fields
		self.replotCount = 0
		self.replotSuspended = True
		self.changeFieldSuspended = False
		self.replot_reset_limits = False
		self.initialView = None
		self.lastSliceIndices = [0, 0, 0] if other is None else list(other.lastSliceIndices)

		QtWidgets.QWidget.__init__(self, parent)
		self.setContentsMargins(2, 2, 2, 2)
		
		self.fig = Figure(figsize=(20,20))
		self.fig.set_tight_layout(None)
		self.fig.set_frameon(False)
		self.cb = None

		self.axes = self.fig.add_subplot(111)
		self.axes.set_xlabel("x")
		self.axes.set_ylabel("y")

		vbox = QtWidgets.QVBoxLayout(self)
		vbox.setContentsMargins(2, 2, 2, 2)
		#vbox.setSpacing(0)

		def makeChangeFieldCallback(index):
			return lambda checked: self.changeField(index, checked)
		
		self.fields = []
		self.currentFieldIndex = other.currentFieldIndex if (other != None) else None

		spacing = 2

		flow = FlowLayout()
		for j, field_group in enumerate(field_groups):
			#gbox = QtWidgets.QHBoxLayout()
			gbox = flow
			for i, field in enumerate(field_group):
				button = QtWidgets.QToolButton()
				field.button = button
				button.setText(field.label)
				if len(field.description):
					button.setToolTip(field.description)
				button.setCheckable(True)
				index = len(self.fields)
				button.toggled.connect(makeChangeFieldCallback(index))
				if i > 0:
					gbox.addSpacing(spacing)
				gbox.addWidget(button)
				self.fields.append(field)
			if j > 0:
				flow.addSpacing(spacing*4)
			#flow.addLayout(gbox)

		flow.addSpacing(spacing*4)
		flow.addStretch()

		hbox = QtWidgets.QHBoxLayout()
		hbox.setAlignment(QtCore.Qt.AlignTop)
		hbox.setSpacing(spacing)

		self.writeVTKButton = QtWidgets.QToolButton()
		self.writeVTKButton.setText("Write VTK")
		self.writeVTKButton.clicked.connect(self.writeVTK)
		hbox.addWidget(self.writeVTKButton)
	
		self.writePNGButton = QtWidgets.QToolButton()
		self.writePNGButton.setText("Write PNG")
		self.writePNGButton.clicked.connect(self.writePNG)
		hbox.addWidget(self.writePNGButton)
		
		self.viewResultDataButton = QtWidgets.QToolButton()
		self.viewResultDataButton.setText("Results")
		self.viewResultDataButton.setCheckable(True)
		self.viewResultDataButton.toggled.connect(self.viewResultData)
		hbox.addWidget(self.viewResultDataButton)

		self.viewXMLButton = QtWidgets.QToolButton()
		self.viewXMLButton.setText("XML")
		self.viewXMLButton.setCheckable(True)
		self.viewXMLButton.toggled.connect(self.viewXML)
		hbox.addWidget(self.viewXMLButton)

		flow.addLayout(hbox)

		vbox.addLayout(flow)

		if len(self.fields) == 0:
			data_shape = [0, 0, 0]
		else:
			data_shape = self.fields[0].data[0].shape[1:4]

		self.sliceCombo = QtWidgets.QComboBox()
		self.sliceCombo.setEditable(False)
		self.sliceCombo.addItem("x")
		self.sliceCombo.addItem("y")
		self.sliceCombo.addItem("z")
		
		if (other != None):
			sliceIndex = other.sliceCombo.currentIndex()
		else:
			sliceIndex = 0
			for i in range(1,3):
				if data_shape[sliceIndex] >= data_shape[i]:
					sliceIndex = i

		self.sliceCombo.lastIndex = sliceIndex
		self.sliceCombo.setCurrentIndex(sliceIndex)
		self.sliceCombo.currentIndexChanged.connect(self.sliceComboChanged)

		self.sliceSlider = QtWidgets.QSlider()
		self.sliceSlider.setOrientation(QtCore.Qt.Horizontal)
		self.sliceSlider.setMinimum(0)
		self.sliceSlider.setMaximum(data_shape[sliceIndex]-1)
		self.sliceSlider.setTickPosition(QtWidgets.QSlider.TicksBothSides)
		self.sliceSlider.setTickInterval(1)
		#self.sliceSlider.sliderMoved.connect(self.sliceSliderChanged)
		if (other != None):
			self.sliceSlider.setValue(int(other.sliceSlider.value()*(self.sliceSlider.maximum()+1)/(other.sliceSlider.maximum()+1)))
		else:
			self.sliceSlider.setValue(int((self.sliceSlider.maximum() + self.sliceSlider.minimum())/2))
		self.sliceSlider.valueChanged.connect(self.sliceSliderChanged)
		self.sliceLabel = QtWidgets.QLabel()
		self.sliceLabel.setText("%s=%04d" % (self.sliceCombo.currentText(), self.sliceSlider.value()))

		hbox1 = QtWidgets.QHBoxLayout()
		hbox1.addWidget(QtWidgets.QLabel("Slice:"))
		hbox1.addWidget(self.sliceCombo)
		hbox1.addWidget(self.sliceSlider)
		hbox1.addWidget(self.sliceLabel)

		self.loadstepSlider = QtWidgets.QSlider()
		self.loadstepSlider.setOrientation(QtCore.Qt.Horizontal)
		self.loadstepSlider.setMinimum(0)
		self.loadstepSlider.setMaximum((len(self.fields[0].data)-1) if len(self.fields) else 0)
		self.loadstepSlider.setTickPosition(QtWidgets.QSlider.TicksBothSides)
		self.loadstepSlider.setTickInterval(1)
		#self.loadstepSlider.sliderMoved.connect(self.loadstepSliderChanged)
		if (other != None):
			self.loadstepSlider.setValue(int(other.loadstepSlider.value()*(self.loadstepSlider.maximum()+1)/(other.loadstepSlider.maximum()+1)))
		else:
			self.loadstepSlider.setValue(self.loadstepSlider.maximum())
		self.loadstepSlider.valueChanged.connect(self.loadstepSliderChanged)
		self.loadstepLabel = QtWidgets.QLabel()
		self.loadstepLabel.setText("%04d" % self.loadstepSlider.value())

		hbox2 = QtWidgets.QHBoxLayout()
		hbox2.addWidget(QtWidgets.QLabel("Loadstep:"))
		hbox2.addWidget(self.loadstepSlider)
		hbox2.addWidget(self.loadstepLabel)

		hbox = QtWidgets.QHBoxLayout()
		hbox.addLayout(hbox1)
		hbox.addLayout(hbox2)
		vbox.addLayout(hbox)

		self.defaultColormap = "coolwarm" if 0 else "jet"
		self.colormapCombo = QtWidgets.QComboBox()
		self.colormapCombo.setEditable(False)
		colormaps = sorted(mcmap.datad, key=lambda s: s.lower())
		for cm in colormaps:
			self.colormapCombo.addItem(cm)
		self.colormapCombo.setCurrentIndex(colormaps.index(self.defaultColormap))
		if (other != None):
			self.colormapCombo.setCurrentIndex(other.colormapCombo.currentIndex())
		self.colormapCombo.currentIndexChanged.connect(self.colormapComboChanged)

		self.alphaSlider = QtWidgets.QSlider()
		self.alphaSlider.setOrientation(QtCore.Qt.Horizontal)
		self.alphaSlider.setMinimum(0)
		self.alphaSlider.setMaximum(10000)
		#self.alphaSlider.setTickPosition(QtWidgets.QSlider.TicksBothSides)
		self.alphaSlider.setTickInterval(1)
		#self.alphaSlider.sliderMoved.connect(self.sliceSliderChanged)
		if (other != None):
			self.alphaSlider.setValue(other.alphaSlider.value())
		self.alphaSlider.valueChanged.connect(self.alphaSliderChanged)
		self.alphaLabel = QtWidgets.QLabel()
		self.alphaLabel.setText("alpha=%4f" % self.getAlpha())

		self.depthViewCheck = QtWidgets.QCheckBox("depth mode")
		if (other != None):
			self.depthViewCheck.setCheckState(other.depthViewCheck.checkState())
		self.depthViewCheck.stateChanged.connect(self.depthViewCheckChanged)

		self.interpolateCheck = QtWidgets.QCheckBox("interpolate")
		if (other != None):
			self.interpolateCheck.setCheckState(other.interpolateCheck.checkState())
		self.interpolateCheck.stateChanged.connect(self.interpolateCheckChanged)

		hbox = QtWidgets.QHBoxLayout()
		hbox.addWidget(QtWidgets.QLabel("Colormap:"))
		hbox.addWidget(self.colormapCombo)
		hbox.addWidget(QtWidgets.QLabel("Contrast:"))
		hbox.addWidget(self.alphaSlider)
		hbox.addWidget(self.alphaLabel)
		hbox.addWidget(self.depthViewCheck)
		hbox.addWidget(self.interpolateCheck)
		vbox.addLayout(hbox)

		self.customBoundsCheck = QtWidgets.QCheckBox("custom")
		if (other != None):
			self.customBoundsCheck.setCheckState(other.customBoundsCheck.checkState())
		self.customBoundsCheck.stateChanged.connect(self.customBoundsCheckChanged)

		self.vminLabel = QtWidgets.QLabel("vmin:")
		self.vminText = QtWidgets.QLineEdit()
		self.vminText.setValidator(QtGui.QDoubleValidator(self.vminText));

		self.vmaxLabel = QtWidgets.QLabel("vmax:")
		self.vmaxText = QtWidgets.QLineEdit()
		self.vmaxText.setValidator(QtGui.QDoubleValidator(self.vmaxText));

		enabled = self.customBoundsCheck.checkState() != 0
		if enabled and other != None:
			self.vminText.setText(other.vminText.text())
			self.vmaxText.setText(other.vmaxText.text())
		self.vminText.setEnabled(enabled)
		self.vmaxText.setEnabled(enabled)

		self.replotButton = QtWidgets.QPushButton("Refresh")
		self.replotButton.clicked.connect(lambda checked: self.replot)

		hbox = QtWidgets.QHBoxLayout()
		hbox.addWidget(QtWidgets.QLabel("Bounds:"))
		hbox.addWidget(self.customBoundsCheck)
		hbox.addWidget(self.vminLabel)
		hbox.addWidget(self.vminText)
		hbox.addWidget(self.vmaxLabel)
		hbox.addWidget(self.vmaxText)
		hbox.addWidget(self.replotButton)
		vbox.addLayout(hbox)

		def makeMaskFieldCallback(index):
			return lambda checked: self.maskField(index, checked)

		materials = field_groups[1]
		if len(materials) >= 2:
			#hbox = QtWidgets.QHBoxLayout()
			hbox.addWidget(QtWidgets.QLabel("Mask:"))
			gbox = QtWidgets.QHBoxLayout()
			for i, field in enumerate(materials):
				button = QtWidgets.QToolButton()
				button.setText(field.label)
				if len(field.description):
					button.setToolTip(field.description)
				button.setCheckable(True)
				button.toggled.connect(makeMaskFieldCallback(index))
				if i > 0:
					gbox.addSpacing(spacing)
				gbox.addWidget(button)
				field.mask_button = button
			hbox.addLayout(gbox)
			#vbox.addLayout(hbox)

		self.stack = QtWidgets.QStackedWidget()
		self.stack.setFrameShape(QtWidgets.QFrame.StyledPanel)
		self.stack.setFrameShadow(QtWidgets.QFrame.Sunken)

		vbox.addWidget(self.stack)
		
		self.figcanvas = FigureCanvas(self.fig)
		self.figcanvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

		#self.figcanvas.setStyle(app.style())
		#self.figcanvas.setStyleSheet("background-color: yellow")

		self.fignavbar = NavigationToolbar(self.figcanvas, self)
		self.fignavbar.set_cursor(cursors.SELECT_REGION)
		self.fignavbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
		self.fignavbar.set_history_buttons = self.setHistoryButtons
		self.fignavbar._update_view_old = self.fignavbar._update_view
		self.fignavbar._update_view = self.fignavbar_update_view
		
		def setIcon(c, names):
			for name in names:
				if QtGui.QIcon.hasThemeIcon(name):
					c.setIcon(QtGui.QIcon.fromTheme(name))
					return True
			return False

		action = self.fignavbar.addAction("Grid")
		action.toggled.connect(lambda c: self.replot())
		action.setCheckable(True)
		setIcon(action, ["view-grid", "show-grid"])
		action.setToolTip("Show grid lines")
		self.fignavbar.insertAction(self.fignavbar.actions()[-8], action)
		self.showGridAction = action

		action = self.fignavbar.addAction("Embed")
		action.triggered.connect(self.saveCurrentView)
		setIcon(action, ["text-xml"])
		action.setToolTip("Embed view into XML document")
		self.fignavbar.insertAction(self.fignavbar.actions()[-2], action)
		
		for c in self.fignavbar.findChildren(QtWidgets.QToolButton):
			if c.text() == "Home":
				setIcon(c, ["go-first-view", "go-home", "go-first", "arrow-left-double"])
			elif c.text() == "Back":
				setIcon(c, ["go-previous-view", "go-previous", "arror-left"])
			elif c.text() == "Forward":
				setIcon(c, ["go-next-view", "go-next", "arror-right"])
			elif c.text() == "Pan":
				#setIcon(c, ["transform-move"])
				pass
			elif c.text() == "Zoom":
				#setIcon(c, ["zoom-select", "zoom-in"])
				pass
			elif c.text() == "Subplots":
				#setIcon(c, [""])
				self.fignavbar.removeAction(c.defaultAction())
			elif c.text() == "Customize":
				#setIcon(c, ["preferences-system"])
				self.fignavbar.removeAction(c.defaultAction())
			elif c.text() == "Save":
				setIcon(c, ["document-save"])

		tb = QtWidgets.QToolBar()
		self.fignavbar.setIconSize(tb.iconSize())
		self.fignavbar.layout().setSpacing(tb.layout().spacing())
		self.fignavbar.sizeHint = tb.sizeHint
		self.fignavbar.locLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
		self.fignavbar.locLabel.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed))
		self.fignavbar.locLabel.setContentsMargins(0,-4,0,-4)
		fm = QtGui.QFontMetrics(self.fignavbar.locLabel.font())
		self.fignavbar.locLabel.setFixedHeight(2*fm.height())

		if other != None:
			if hasattr(self.fignavbar, "_views"):
				self.fignavbar._views = copy.copy(other.fignavbar._views)
				self.fignavbar._positions = copy.copy(other.fignavbar._positions)
			else:
				# TODO: copy is not possible due to axis reference
				pass
				"""
				for e in other.fignavbar._nav_stack._elements:
					
				self.fignavbar._nav_stack = copy.copy(other.fignavbar._nav_stack)
						self.fignavbar._nav_stack.push(
							WeakKeyDictionary(
								{ax: (views[i], pos[i])
							for i, ax in enumerate(self.figcanvas.figure.get_axes())}))
				"""

		wvbox = QtWidgets.QVBoxLayout()
		wvbox.setContentsMargins(2, 2, 2, 2)
		#wvbox.setSpacing(0)
		wvbox.addWidget(self.fignavbar)
		wvbox.addWidget(self.figcanvas)
		wrap = QtWidgets.QWidget()
		wrap.setStyleSheet("background-color:%s;" % pal.base().color().name());
		wrap.setLayout(wvbox)
		self.stack.addWidget(wrap)

		self.textEdit = XMLTextEdit()
		self.textEdit.setReadOnly(True)
		self.textEdit.setPlainText(xml)
		self.textEdit.setFrameShape(QtWidgets.QFrame.NoFrame)
		self.textEdit.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
		self.stack.addWidget(self.textEdit)

		if not app.pargs.disable_browser:
			self.resultTextEdit = QtWebKitWidgets.QWebView()
			self.resultPage = MyWebPage()
			self.resultPage.setHtml(defaultCSS() + resultText)
			self.resultTextEdit.setPage(self.resultPage)
		else:
			self.resultTextEdit = QtWidgets.QTextEdit()
			self.resultTextEdit.setReadOnly(True)
			self.resultTextEdit.setHtml(resultText)
		self.resultTextEdit.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
		self.stack.addWidget(self.resultTextEdit)

		if other != None:
			self.viewXMLButton.setChecked(other.viewXMLButton.isChecked())
			self.viewResultDataButton.setChecked(other.viewResultDataButton.isChecked())

		self.setLayout(vbox)

		if len(self.fields) == 0:
			self.currentFieldIndex = None
		elif self.currentFieldIndex is None:
			self.currentFieldIndex = 0

		if not self.currentFieldIndex is None:
			if (self.currentFieldIndex >= len(self.fields)):
				self.currentFieldIndex = 0
			self.fields[self.currentFieldIndex].button.setChecked(True)
	
		try:
			if not xml_root is None:
				view = xml_root.find('view')
				if not view is None:
					self.setViewXML(view)
		except:
			print(traceback.format_exc())

		data = self.getCurrentSlice()
		self.resetBounds(data)
		self.replotSuspended = False
		self.replot(data)

		self.updateFigCanvasVisible()

	def setHistoryButtons(self):
		#self.redrawCanvas()
		pass

	def getViewXML(self):

		view = ET.Element('view')

		field = ET.SubElement(view, 'field')
		field.text = self.fields[self.currentFieldIndex].key

		if self.viewXMLButton.isChecked():
			vmode = 'xml'
		elif self.viewResultDataButton.isChecked():
			vmode = 'results'
		else:
			vmode = 'plot'
		if vmode != "plot":
			mode = ET.SubElement(view, 'mode')
			mode.text = vmode

		slice_dim = ET.SubElement(view, 'slice_dim')
		slice_dim.text = self.sliceCombo.currentText()

		if self.sliceSlider.minimum() < self.sliceSlider.maximum():
			slice_index = ET.SubElement(view, 'slice_index')
			slice_index.text = str((self.sliceSlider.value()+0.5)/(self.sliceSlider.maximum()+1))

		if self.loadstepSlider.minimum() < self.loadstepSlider.maximum():
			loadstep = ET.SubElement(view, 'loadstep')
			loadstep.text = str((self.loadstepSlider.value()+0.5)/(self.loadstepSlider.maximum()+1))

		if self.colormapCombo.currentText() != self.defaultColormap:
			colormap = ET.SubElement(view, 'colormap')
			colormap.text = self.colormapCombo.currentText()

		valpha = self.getAlpha()
		if valpha != 0.0:
			alpha = ET.SubElement(view, 'alpha')
			alpha.text = str(valpha)

		vinterpolate = 1 if self.interpolateCheck.checkState() else 0
		if vinterpolate != 0:
			interpolate = ET.SubElement(view, 'interpolate')
			interpolate.text = str(vinterpolate)

		vdepth_view = 1 if self.depthViewCheck.checkState() else 0
		if vdepth_view != 0:
			depth_view = ET.SubElement(view, 'depth_view')
			depth_view.text = str(vdepth_view)

		vcustom_bounds = 1 if self.customBoundsCheck.checkState() else 0
		if vcustom_bounds != 0:
			custom_bounds = ET.SubElement(view, 'custom_bounds')
			custom_bounds.text = str(vcustom_bounds)
			vmin = ET.SubElement(view, 'vmin')
			vmin.text = self.vminText.text()
			vmax = ET.SubElement(view, 'vmax')
			vmax.text = self.vmaxText.text()

		if hasattr(self.fignavbar, "_views"):
			views = self.fignavbar._views()
		else:
			ns = self.fignavbar._nav_stack()
			if not ns is None:
				arr = ns.values()
			views = [n[0] for n in arr]

		if not views is None and len(views):
			v = views[0]
			data = self.getCurrentSlice()
			numcols, numrows = data.shape
			norm = [numcols, numcols, numrows, numrows]
			defaults = [0.0, 1.0, 0.0, 1.0]
			values = [(v[i] + 0.5)/norm[i] for i in range(4)]
			if values != defaults:
				for i, key in enumerate(["zoom_xmin", "zoom_xmax", "zoom_ymin", "zoom_ymax"]):
					vi = ET.SubElement(view, key)
					vi.text = str(values[i])

		if len(self.view_extra_fields):
			extra_fields = ET.SubElement(view, 'extra_fields')
			extra_fields.text = ",".join(self.view_extra_fields)

		# indent XML
		indent = "\t"
		view.text = "\n" + indent
		for e in view:
			e.tail = "\n" + indent
		e.tail = "\n"

		return view;

	def saveCurrentView(self):
		app = QtWidgets.QApplication.instance()
		xml = app.window.textEdit.toPlainText()
		view = self.getViewXML()
		sub = ET.tostring(view, encoding='unicode')
		lines = sub.split("\n")
		indent = "\t"
		for i in range(len(lines)):
			lines[i] = indent + lines[i]
		sub = "\n".join(lines)
		match = re.search("\s*<view>.*</view>\s*", xml, flags=re.S)
		pre = "\n\n"
		post = "\n\n"
		if not match:
			match = re.search("\s*</settings>", xml)
			post = "\n\n</settings>"
		if match:
			c = app.window.textEdit.textCursor()
			c.setPosition(match.start())
			c.movePosition(QtGui.QTextCursor.Right, QtGui.QTextCursor.KeepAnchor, match.end()-match.start())
			c.insertText(pre + sub + post)
			c.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.MoveAnchor, len(post))
			c.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.KeepAnchor, len(sub))
			app.window.textEdit.setTextCursor(c)

	def setViewXML(self, view):

		field = view.find('field')
		if not field is None:
			for f in self.fields:
				if f.key == field.text:
					f.button.setChecked(True)
					break

		mode = view.find('mode')
		if not mode is None:
			if mode.text == 'xml':
				self.viewXMLButton.setChecked(True)
			elif mode.text == 'results':
				self.viewResultDataButton.setChecked(True)
		
		slice_dim = view.find('slice_dim')
		if not slice_dim is None:
			self.sliceCombo.setCurrentText(slice_dim.text)

		slice_index = view.find('slice_index')
		if not slice_index is None:
			self.sliceSlider.setValue(int(float(slice_index.text)*(self.sliceSlider.maximum()+1)))

		loadstep = view.find('loadstep')
		if not loadstep is None:
			self.loadstepSlider.setValue(int(float(loadstep.text)*(self.loadstepSlider.maximum()+1)))

		colormap = view.find('colormap')
		if not colormap is None:
			self.colormapCombo.setCurrentText(colormap.text)

		alpha = view.find('alpha')
		if not alpha is None:
			self.setAlpha(float(alpha.text))

		interpolate = view.find('interpolate')
		if not interpolate is None:
			vinterpolate = int(interpolate.text) != 0
			self.interpolateCheck.setChecked(vinterpolate)

		depth_view = view.find('depth_view')
		if not depth_view is None:
			vdepth_view = int(depth_view.text) != 0
			self.depthViewCheck.setChecked(vdepth_view)

		custom_bounds = view.find('custom_bounds')
		if not custom_bounds is None:
			vcustom_bounds = int(custom_bounds.text) != 0
			self.customBoundsCheck.setChecked(vcustom_bounds)
			if vcustom_bounds:
				vmin = view.find('vmin')
				if not vmin is None:
					self.vminText.setText(vmin.text)

				vmax = view.find('vmax')
				if not vmax is None:
					self.vmaxText.setText(vmax.text)

		zoom = np.zeros(4)
		data = self.getCurrentSlice()
		if not data is None:
			numcols, numrows = data.shape
			norm = [numcols, numcols, numrows, numrows]
			for i, key in enumerate(["zoom_xmin", "zoom_xmax", "zoom_ymin", "zoom_ymax"]):
				val = view.find(key)
				if not val is None:
					zoom[i] = float(val.text)*norm[i] - 0.5
				else:
					zoom = None
					break

		self.initialView = zoom

	def writeVTK(self):
		
		filename, _filter = QtWidgets.QFileDialog.getSaveFileName(self, "Save VTK", os.getcwd(), "VTK Files (*.vtk)")
		if (filename == ""):
			return
		
		w = WriteVTKWidget(filename, self.rve_dims, self.field_groups, parent=self)
		w.exec_()


	def writePNG(self):
		
		if (self.currentFieldIndex == None):
			return
		
		filename, _filter = QtWidgets.QFileDialog.getSaveFileName(self, "Save PNG", os.getcwd(), "PNG Files (*.png)")
		if (filename == ""):
			return
		filename, ext = os.path.splitext(filename)
		filename += ".png"
		
		vmin = float(self.vminText.text())
		vmax = float(self.vmaxText.text())
		cm = mcmap.get_cmap(self.colormapCombo.currentText(), 2**13)

		data = self.getCurrentSlice()
		data = np.rot90(data)
		data = (data-vmin)/(vmax-vmin)
		data = cm(data)

		image = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
		image[:,:,0:3] = data[:,:,0:3]*255

		img = scipy.misc.toimage(image, high=np.max(image), low=np.min(image))
		img.save(filename, "PNG", compress_level=0)

		
		template_src = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../gui/plot_template.tex")
		template_dest, ext = os.path.splitext(filename)
		template_dest += ".tex"
		
		with open(template_src, "rt") as f:
			template = f.read()


		colormap = ""
		for c in np.linspace(0.0, 1.0, 256):
			color = cm(c)
			colormap += "rgb255=(%d, %d, %d);\n" % (color[0]*255.99, color[1]*255.99, color[2]*255.99)

		#rgb255(1000pt)=(0, 255, 255)

		labels = self.cb.ax.get_yticklabels()
		labels = [l._text.replace("\u2212", "-") for l in labels]
		zticks = ",".join(labels)
		zticklabels = ",".join(["$%s$" % l for l in labels])

		field = self.fields[self.currentFieldIndex]
		zlabel = field.label
		zlabel = zlabel.replace(u"ε", r'\varepsilon')
		zlabel = zlabel.replace(u"σ", r'\sigma')
		zlabel = zlabel.replace(u"φ", r'\varphi')
		zlabel = zlabel.replace(u"∇", r'\nabla')
		if ("_" in zlabel):
			zlabel = zlabel.replace("_", '_{') + "}"
		zlabel = "$" + zlabel + "$"

		template = template.replace("#filename", os.path.basename(filename))
		template = template.replace("#xmax", str(data.shape[1]))
		template = template.replace("#ymax", str(data.shape[0]))
		template = template.replace("#zmin", str(vmin))
		template = template.replace("#zmax", str(vmax))
		template = template.replace("#zlabel", zlabel)
		template = template.replace("#zticks", zticks)
		template = template.replace("#zticklabels", zticklabels)
		template = template.replace("#colormap", colormap)

		with codecs.open(template_dest, mode="w", encoding="utf-8") as f:
			f.write(template)

		if False:
			subprocess.call(["pdflatex", template_dest], cwd=os.path.dirname(filename))

			pdf, ext = os.path.splitext(filename)
			pdf += ".pdf"
			subprocess.Popen(["okular", pdf])

		#scipy.misc.imsave(filename, image) #, 'PNG')

	def viewResultData(self, state):
		v = (state == 0)
		self.viewXMLButton.setChecked(False)
		self.viewResultDataButton.setChecked(not v)
		self.updateFigCanvasVisible()
		
	def viewXML(self, state):
		v = (state == 0)
		self.viewResultDataButton.setChecked(False)
		self.viewXMLButton.setChecked(not v)
		self.updateFigCanvasVisible()

	def updateFigCanvasVisible(self):
		if self.viewXMLButton.isChecked():
			self.stack.setCurrentIndex(1)
		elif self.viewResultDataButton.isChecked():
			self.stack.setCurrentIndex(2)
		else:
			self.stack.setCurrentIndex(0)
	
	def colormapComboChanged(self, index):
		self.replot()
	
	def loadstepSliderChanged(self):
		data = self.getCurrentSlice()
		self.resetBounds(data)
		self.replot(data)
		self.loadstepLabel.setText("%04d" % self.loadstepSlider.value())

	def sliceComboChanged(self, index):
		rs = self.replotSuspended
		self.replotSuspended = True
		self.lastSliceIndices[self.sliceCombo.lastIndex] = self.sliceSlider.value()
		self.sliceCombo.lastIndex = index
		self.replot_reset_limits = True
		data_shape = self.fields[0].data[0].shape[1:4]
		self.sliceSlider.setMaximum(data_shape[index]-1)
		self.sliceSlider.setValue(self.lastSliceIndices[index])
		self.replotSuspended = rs
		self.sliceSliderChanged()

	def depthViewCheckChanged(self, state):
		self.replot()

	def interpolateCheckChanged(self, state):
		self.replot()

	def customBoundsCheckChanged(self, state):
		enable = (state != 0)
		self.vminText.setEnabled(enable)
		self.vmaxText.setEnabled(enable)
		data = self.getCurrentSlice()
		self.resetBounds(data)
		if (state == 0):
			self.replot(data)

	def setAlpha(self, a):
		a_max = 0.4999
		a = max(min(a, a_max), 0.0)
		self.alphaSlider.setValue(int(self.alphaSlider.maximum()*(a/a_max)**(1.0/3.0) + 0.5))
	
	def getAlpha(self):
		return 0.4999*(self.alphaSlider.value()/self.alphaSlider.maximum())**3
	
	def getCurrentSlice(self):
		if self.currentFieldIndex is None:
			return None

		#print("getCurrentSlice")
		#traceback.print_stack()

		field = self.fields[self.currentFieldIndex]
		s_index = self.sliceSlider.value()
		ls_index = self.loadstepSlider.value()
		sliceIndex = self.sliceCombo.currentIndex()
		depth = 1
		depth_max = self.sliceSlider.maximum() - s_index + 1

		if field.name == "phi" and self.depthViewCheck.isChecked():
			depth = 1 + self.sliceSlider.maximum()

		s_index_end = s_index + min(depth, depth_max)

		# get mask
		mask_fields = []
		materials = self.field_groups[1]
		if len(materials) >= 2:
			for i, mfield in enumerate(materials):
				if mfield.mask_button.isChecked():
					mask_fields.append(mfield)

		if (sliceIndex == 0):
			data = field.data[ls_index][field.component,s_index:s_index_end,:,:]
			mask = np.zeros_like(data) if len(mask_fields) else 1.0
			for f in mask_fields:
				mask = np.maximum(mask, f.data[ls_index][f.component,s_index:s_index_end,:,:])
		elif (sliceIndex == 1):
			data = field.data[ls_index][field.component,:,s_index:s_index_end,:]
			mask = np.zeros_like(data) if len(mask_fields) else 1.0
			for f in mask_fields:
				mask = np.maximum(mask, f.data[ls_index][f.component,:,s_index:s_index_end,:])
		else:
			data = field.data[ls_index][field.component,:,:,s_index:s_index_end]
			mask = np.zeros_like(data) if len(mask_fields) else 1.0
			for f in mask_fields:
				mask = np.maximum(mask, f.data[ls_index][f.component,:,:,s_index:s_index_end])

		if depth >= 1:
			z = np.indices(data.shape)[sliceIndex]
			data = np.max(data*np.exp((-3.0/depth)*z), axis=sliceIndex)
			if len(mask_fields):
				mask = np.max(mask, axis=sliceIndex)
		#else:
		#	data = np.squeeze(data, axis=sliceIndex)

		mask = mask == 1.0

		return mask*data

	def resetBounds(self, data=None):
		if (self.customBoundsCheck.checkState() == 0 and self.currentFieldIndex != None):

			if data is None:
				data = self.getCurrentSlice()
			
			alpha = self.getAlpha()
			vmin = np.amin(data)
			vmax = np.amax(data)

			if (alpha > 0):
				nbins = self.alphaSlider.maximum()
				hist, bins = np.histogram(data, range=(vmin, vmax), bins=nbins, density=True)
				dx = (vmax-vmin)/nbins

				s = 0
				for i in range(nbins):
					if (s >= alpha):
						vmin = bins[i]
						break
					s += hist[i]*dx
				
				s = 0
				for i in reversed(range(nbins)):
					if (s >= alpha):
						vmax = bins[i]
						break
					s += hist[i]*dx

			self.vminText.setText(str(vmin))
			self.vmaxText.setText(str(vmax))

	def maskField(self, index, checked):
		data = self.getCurrentSlice()
		self.resetBounds(data)
		self.replot(data)

	def changeField(self, index, checked):

		if (self.changeFieldSuspended):
			return

		data = None

		if checked:
			self.currentFieldIndex = index
			self.changeFieldSuspended = True
			for i, field in enumerate(self.fields):
				field.button.setChecked(i == index)
			self.changeFieldSuspended = False
			data = self.getCurrentSlice()
			self.resetBounds(data)
			self.viewXMLButton.setChecked(False)
			self.viewResultDataButton.setChecked(False)
		else:
			self.currentFieldIndex = None

		self.replot(data)

	def sliceSliderChanged(self):
		data = self.getCurrentSlice()
		self.resetBounds(data)
		self.replot(data)
		self.sliceLabel.setText("%s=%04d" % (self.sliceCombo.currentText(), self.sliceSlider.value()))

	def alphaSliderChanged(self):
		self.alphaLabel.setText("alpha=%4f" % self.getAlpha())
		if self.customBoundsCheck.checkState() != 0:
			self.customBoundsCheck.setCheckState(0)
		else:
			self.customBoundsCheckChanged(0)

	def replot(self, data=None):

		if self.replotSuspended:
			return

		#print("replot")
		#traceback.print_stack()

		self.replotCount += 1

		xlim = None
		ylim = None

		firstReplot = self.cb is None

		if not firstReplot:
			if not self.replot_reset_limits:
				xlim = self.axes.get_xlim()
				ylim = self.axes.get_ylim()
			self.cb.ax.clear()
			cbax = self.cb.ax
		else:
			cbax = None
		
		self.axes.clear()

		if (self.currentFieldIndex != None):
			
			if data is None:
				data = self.getCurrentSlice()

			s_index = self.sliceCombo.currentIndex()
			coords = ["x", "y", "z"]
			z_cord = coords[s_index]
			xy_cord = coords
			del xy_cord[s_index]

			vmin = float(self.vminText.text())
			vmax = float(self.vmaxText.text())
			
			# NOTE: interpolation is matplotlib 2.0 is still buggy
			# https://github.com/matplotlib/matplotlib/issues/8631
			#methods = ['bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
			interpolation = "bicubic" if self.interpolateCheck.isChecked() else "nearest"

			field_name = self.fields[self.currentFieldIndex].name
			num_colors = self.fields[self.currentFieldIndex].num_discrete_values
			value_labels = self.fields[self.currentFieldIndex].value_labels
			if np.isinf(num_colors):
				num_colors = 2**11

			if field_name == "phi":
				vmin = 0.0
				vmax = 1.0

			if len(value_labels) > 0:
				vmin = 0
				vmax = len(value_labels)
				interpolation = "nearest"

			#color_norm = matplotlib.colors.SymLogNorm(linthresh=1e-2, linscale=1)
			cm = mcmap.get_cmap(self.colormapCombo.currentText(), num_colors)

			p = self.axes.imshow(data.T, interpolation=interpolation, origin="lower",
				norm=None, cmap=cm, vmin=vmin, vmax=vmax)

			if (cbax != None):
				self.cb = self.fig.colorbar(p, cax=cbax)
			else:
				self.cb = self.fig.colorbar(p, shrink=0.7, pad=0.05, fraction=0.1, use_gridspec=True)
		
			if len(value_labels) > 0:
				self.cb.set_ticks(list(value_labels.keys()))
				self.cb.set_ticklabels(list(value_labels.values()))

			z_label = self.fields[self.currentFieldIndex].label
			self.cb.ax.set_title(z_label, y=1.03)

			numcols, numrows = data.shape
			def format_coord(x, y):
				col = int(x+0.5)
				row = int(y+0.5)
				s = '%s = %d, %s = %d' % (xy_cord[0], col, xy_cord[1], row)
				if col>=0 and col<numcols and row>=0 and row<numrows:
					z = data[col,row]
					return '%s\n%s = %.8g' % (s, z_label, z)
				else:
					return s
			def get_cursor_data(event):
				return None
			self.axes.format_coord = format_coord
			p.get_cursor_data = get_cursor_data

			font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.GeneralFont)
			self.axes.set_xlabel(xy_cord[0], labelpad=font.pointSize())
			self.axes.set_ylabel(xy_cord[1], labelpad=font.pointSize())

			# show grid
			if self.showGridAction.isChecked():
				b = 1
				self.axes.set_xticks(np.arange(-0.5+b, numcols-b, 1), minor=True);
				self.axes.set_yticks(np.arange(-0.5+b, numrows-b, 1), minor=True);
				self.axes.grid(which='minor', color='w', linestyle='-', alpha=0.5, linewidth=0.5, antialiased=False, snap=True)

			if (xlim == None):
				xlim = [0, numcols-1]
			if (ylim == None):
				ylim = [0, numrows-1]
			
			self.figcanvas.setVisible(True)
			self.fignavbar.setVisible(True)
		else:
			self.figcanvas.setVisible(False)
			self.fignavbar.setVisible(False)

		if (xlim != None):
			self.axes.set_xlim(xlim)
		if (ylim != None):
			self.axes.set_ylim(ylim)

		"""
		self.axes.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
		self.axes.yaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
		"""

		if self.replot_reset_limits:
			self.fignavbar.update()

		if firstReplot:
			self.figcanvas.draw()

			if not self.initialView is None:
				# /usr/lib/python3/dist-packages/matplotlib/backend_bases.py
				views = []
				pos = []
				for a in self.figcanvas.figure.get_axes():
					views.append(a._get_view())
					pos.append((
						a.get_position(True).frozen(),
						a.get_position().frozen()))
				if len(views):
					if hasattr(self.fignavbar, "_views"):
						self.fignavbar._views.push(views)
						self.fignavbar._positions.push(pos)
						views = copy.copy(views)
						views[0] = tuple(self.initialView)
						self.fignavbar._views.push(views)
						self.fignavbar._positions.push(pos)
						self.fignavbar._views._pos = len(self.fignavbar._views._elements)-1
						self.fignavbar._positions._pos = len(self.fignavbar._positions._elements)-1
					else:
						self.fignavbar._nav_stack.push(
							WeakKeyDictionary(
								{ax: (views[i], pos[i])
							for i, ax in enumerate(self.figcanvas.figure.get_axes())}))
						views = copy.copy(views)
						views[0] = tuple(self.initialView)
						self.fignavbar._nav_stack.push(
							WeakKeyDictionary(
								{ax: (views[i], pos[i])
							for i, ax in enumerate(self.figcanvas.figure.get_axes())}))
						self.fignavbar._nav_stack._pos = len(self.fignavbar._nav_stack._elements)-1

		self.fignavbar_update_view()
		self.replot_reset_limits = False

	def fignavbar_update_view(self):
		
		self.fignavbar._update_view_old()

		s = self.figcanvas.size()
		s.setWidth(s.width()+0)
		e = QtGui.QResizeEvent(s, self.figcanvas.size())
		self.figcanvas.resizeEvent(e)

		self.figcanvas.draw()

		#print("update")

	def redrawCanvas(self):

		self.figcanvas.draw()

		if hasattr(self.fignavbar, "_views"):
			views = self.fignavbar._views()
			if not views is None:
				pos = self.fignavbar._positions()
				for i, a in enumerate(self.figcanvas.figure.get_axes()):
					a._set_view(views[i])
					# Restore both the original and modified positions
					a.set_position(pos[i][0], 'original')
					a.set_position(pos[i][1], 'active')
					#a.reset_position()
		else:
			views = self.fignavbar._nav_stack()
			if not views is None:
				items = list(views.items())
				for ax, (view, (pos_orig, pos_active)) in items:
					ax._set_view(view)
					# Restore both the original and modified positions
					ax.set_position(pos_orig, 'original')
					ax.set_position(pos_active, 'active')
					#ax.reset_position()


class XMLHighlighter(QtGui.QSyntaxHighlighter):

	def __init__(self, parent=None):
		super(XMLHighlighter, self).__init__(parent)

		self.highlightingRules = []

		app = QtWidgets.QApplication.instance()
		pal = app.palette()

		xmlElementFormat = QtGui.QTextCharFormat()
		xmlElementFormat.setFontWeight(QtGui.QFont.Bold)
		xmlElementFormat.setForeground(QtCore.Qt.darkGreen)
		self.highlightingRules.append((QtCore.QRegExp("<[/\s]*[A-Za-z0-9_-]+[\s/>]+"), xmlElementFormat))

		keywordFormat = QtGui.QTextCharFormat()
		keywordFormat.setFontWeight(QtGui.QFont.Bold)
		keywordFormat.setForeground(QtCore.Qt.gray)
		keywordPatterns = ["[/?]*>", "<([?]xml)?", "=", "['\"]"]
		self.highlightingRules += [(QtCore.QRegExp(pattern), keywordFormat)
				for pattern in keywordPatterns]

		xmlAttributeFormat = QtGui.QTextCharFormat()
		xmlAttributeFormat.setFontWeight(QtGui.QFont.Bold)
		#xmlAttributeFormat.setFontItalic(True)
		xmlAttributeFormat.setForeground(pal.link().color())
		self.highlightingRules.append((QtCore.QRegExp("\\b[A-Za-z0-9_-]+(?=\\=)"), xmlAttributeFormat))

		valueFormat = QtGui.QTextCharFormat()
		valueFormat.setForeground(pal.windowText().color())
		self.highlightingRules.append((QtCore.QRegExp("['\"][^'\"]*['\"]"), valueFormat))

		self.commentFormat = QtGui.QTextCharFormat()
		self.commentFormat.setForeground(QtCore.Qt.gray)
		self.commentStartExpression = QtCore.QRegExp("<!--")
		self.commentEndExpression = QtCore.QRegExp("-->")

		self.pythonStartExpression = QtCore.QRegExp("<python>")
		self.pythonEndExpression = QtCore.QRegExp("</python>")
		self.highlightingRulesPython = []

		self.pythonDefaultFormat = QtGui.QTextCharFormat()

		keywordFormat = QtGui.QTextCharFormat()
		keywordFormat.setFontWeight(QtGui.QFont.Bold)
		keywordFormat.setForeground(QtCore.Qt.darkYellow)
		#keywordFormat.setTextOutline(QtGui.QPen(QtCore.Qt.white))
		self.highlightingRulesPython.append((QtCore.QRegExp(
			"\\b(" + "|".join(keyword.kwlist) + ")\\b"), keywordFormat, 0))
		self.highlightingRulesPython.append((QtCore.QRegExp(
			"(^|\s+|[^\w.]+)(" + "|".join(list(globals()['__builtins__'])) + ")\\s*\("), keywordFormat, 2))

		keywordFormat = QtGui.QTextCharFormat()
		keywordFormat.setFontWeight(QtGui.QFont.Bold)
		keywordFormat.setForeground(QtCore.Qt.gray)
		self.highlightingRulesPython.append((QtCore.QRegExp("[+-*/=%<>!,()\\[\\]{}.\"']+"), keywordFormat, 0))

		commentFormat = QtGui.QTextCharFormat()
		commentFormat.setForeground(QtCore.Qt.gray)
		self.highlightingRulesPython.append((QtCore.QRegExp("#.*"), commentFormat, 0))

		self.pythonOnly = False
		if self.pythonOnly:
			self.pythonStartExpression = QtCore.QRegExp("^")
			self.pythonEndExpression = QtCore.QRegExp("$")


	def highlightBlock(self, text):
		
		if not self.pythonOnly:

			#for every pattern
			for pattern, format in self.highlightingRules:

				#Check what index that expression occurs at with the ENTIRE text
				index = pattern.indexIn(text)

				#While the index is greater than 0
				while index >= 0:

					#Get the length of how long the expression is true, set the format from the start to the length with the text format
					length = pattern.matchedLength()
					self.setFormat(index, length, format)

					#Set index to where the expression ends in the text
					index = pattern.indexIn(text, index + length)

		Flag_Comment = 1
		Flag_Python = 2
		state = 0

		# handle python

		startIndex = 0
		if max(self.previousBlockState(), 0) & Flag_Python == 0:
			# means we are not in a comment
			startIndex = self.pythonStartExpression.indexIn(text)
			if startIndex >= 0:
				startIndex += 8
		
		while startIndex >= 0:
			endIndex = self.pythonEndExpression.indexIn(text, startIndex)
			pythonLength = 0
			if endIndex == -1:
				# means block is python code
				state = state | Flag_Python
				endIndex = len(text)
			
			# format python
			self.setFormat(startIndex, endIndex-startIndex, self.pythonDefaultFormat)

			#for every pattern
			for pattern, format, matchIndex in self.highlightingRulesPython:
	 
				#Check what index that expression occurs at with the ENTIRE text
				index = pattern.indexIn(text, startIndex)

				while index >= startIndex and index <= endIndex:

					texts = pattern.capturedTexts()
					for i in range(1, matchIndex):
						index += len(texts[i])

					length = len(texts[matchIndex])
					self.setFormat(index, length, format)
	 
					#Set index to where the expression ends in the text
					index = pattern.indexIn(text, index + length)

			startIndex = self.pythonStartExpression.indexIn(text, endIndex + 9)
			if startIndex >= 0:
				startIndex += 8

		# handle comments

		startIndex = 0
		if max(self.previousBlockState(), 0) & Flag_Comment == 0:
			# means we are not in a comment
			startIndex = self.commentStartExpression.indexIn(text)
		
		while startIndex >= 0:
			endIndex = self.commentEndExpression.indexIn(text, startIndex)
			commentLength = 0
			if endIndex == -1:
				# means block is a comment
				state = state | Flag_Comment
				commentLength = len(text) - startIndex
			else:
				commentLength = endIndex - startIndex + self.commentEndExpression.matchedLength()
			self.setFormat(startIndex, commentLength, self.commentFormat)
			startIndex = self.commentStartExpression.indexIn(text, startIndex + commentLength)

		self.setCurrentBlockState(state)



class XMLTextEdit(QtWidgets.QTextEdit):

	def __init__(self, parent = None):
		QtWidgets.QTextEdit.__init__(self, parent)

		app = QtWidgets.QApplication.instance()

		doc = QtGui.QTextDocument()
		option = QtGui.QTextOption()
		option.setFlags(QtGui.QTextOption.ShowLineAndParagraphSeparators | QtGui.QTextOption.ShowTabsAndSpaces)
		#doc.setDefaultTextOption(option)
		self.setDocument(doc)

		font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
		fontFamily = app.settings.value("fontFamily", "")
		if fontFamily != "":
			font = QtGui.QFont(fontFamily)
		
		font.setFixedPitch(True)
		font.setPointSize(int(app.settings.value("fontPointSize", font.pointSize())))
		self.setFont(font)
		fontmetrics = QtGui.QFontMetrics(font)
		self.setTabStopWidth(int(app.settings.value("tabStopWidth", 2*fontmetrics.width(' '))))
		self.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
		self.setAcceptRichText(False)

		# add syntax highlighting
		self.highlighter = XMLHighlighter(self.document())

	def keyPressEvent(self, e):

		if e.key() == QtCore.Qt.Key_Tab:
			if (e.modifiers() == QtCore.Qt.ControlModifier) and self.decreaseSelectionIndent():
				return
			if self.increaseSelectionIndent():
				return
		if e.key() in [QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter]:
			if self.insertNewLine():
				return

		QtWidgets.QTextEdit.keyPressEvent(self, e)

	def insertNewLine(self):

		curs = self.textCursor()

		#if curs.hasSelection() or not curs.atBlockEnd():
		#	return False

		line = curs.block().text().rstrip()
		indent = line[0:(len(line) - len(line.lstrip()))]

		if len(line) > 2:
			if line[-1] == ">":
				for i in range(2, len(line)):
					if line[-i] == "<":
						indent += "\t"
						break
					if line[-i] == "/":
						break
			if line[-1] == ":":
				indent += "\t"

		curs.insertText("\n" + indent)
		self.setTextCursor(curs)
		return True

	def decreaseSelectionIndent(self):
		
		curs = self.textCursor()

		# Do nothing if we don't have a selection.
		if not curs.hasSelection():
			return False

		# Get the first and count of lines to indent.

		spos = curs.anchor()
		epos = curs.position()

		if spos > epos:
			hold = spos
			spos = epos
			epos = hold

		curs.setPosition(spos, QtGui.QTextCursor.MoveAnchor)
		sblock = curs.block().blockNumber()

		curs.setPosition(epos, QtGui.QTextCursor.MoveAnchor)
		eblock = curs.block().blockNumber()

		# Do the indent.

		curs.setPosition(spos, QtGui.QTextCursor.MoveAnchor)
		curs.beginEditBlock()

		for i in range(eblock - sblock + 1):
			curs.movePosition(QtGui.QTextCursor.StartOfBlock, QtGui.QTextCursor.MoveAnchor)
			curs.movePosition(QtGui.QTextCursor.Right, QtGui.QTextCursor.KeepAnchor, 1)
			if curs.selectedText() in ["\t", " "]:
				curs.removeSelectedText()
			curs.movePosition(QtGui.QTextCursor.NextBlock, QtGui.QTextCursor.MoveAnchor)

		curs.endEditBlock()

		# Set our cursor's selection to span all of the involved lines.

		curs.setPosition(spos, QtGui.QTextCursor.MoveAnchor)
		curs.movePosition(QtGui.QTextCursor.StartOfBlock, QtGui.QTextCursor.MoveAnchor)

		while (curs.block().blockNumber() < eblock):
			curs.movePosition(QtGui.QTextCursor.NextBlock, QtGui.QTextCursor.KeepAnchor)

		curs.movePosition(QtGui.QTextCursor.EndOfBlock, QtGui.QTextCursor.KeepAnchor)

		# Done!
		self.setTextCursor(curs)

		return True

	def increaseSelectionIndent(self):

		curs = self.textCursor()

		# Do nothing if we don't have a selection.
		if not curs.hasSelection():
			return False

		# Get the first and count of lines to indent.

		spos = curs.anchor()
		epos = curs.position()

		if spos > epos:
			hold = spos
			spos = epos
			epos = hold

		curs.setPosition(spos, QtGui.QTextCursor.MoveAnchor)
		sblock = curs.block().blockNumber()

		curs.setPosition(epos, QtGui.QTextCursor.MoveAnchor)
		eblock = curs.block().blockNumber()

		# Do the indent.

		curs.setPosition(spos, QtGui.QTextCursor.MoveAnchor)
		curs.beginEditBlock()

		for i in range(eblock - sblock + 1):
			curs.movePosition(QtGui.QTextCursor.StartOfBlock, QtGui.QTextCursor.MoveAnchor)
			curs.insertText("\t")
			curs.movePosition(QtGui.QTextCursor.NextBlock, QtGui.QTextCursor.MoveAnchor)

		curs.endEditBlock()

		# Set our cursor's selection to span all of the involved lines.

		curs.setPosition(spos, QtGui.QTextCursor.MoveAnchor)
		curs.movePosition(QtGui.QTextCursor.StartOfBlock, QtGui.QTextCursor.MoveAnchor)

		while (curs.block().blockNumber() < eblock):
			curs.movePosition(QtGui.QTextCursor.NextBlock, QtGui.QTextCursor.KeepAnchor)

		curs.movePosition(QtGui.QTextCursor.EndOfBlock, QtGui.QTextCursor.KeepAnchor)

		# Done!
		self.setTextCursor(curs)

		return True


class HelpWidgetCommon(QtCore.QObject):

	updateHtml = QtCore.pyqtSignal('QString')

	def __init__(self, editor):

		super(QtCore.QObject, self).__init__(editor)

		self.editor = editor
		self.editor.selectionChanged.connect(self.editorSelectionChanged)
		self.editor.textChanged.connect(self.editorSelectionChanged)
		self.editor.cursorPositionChanged.connect(self.editorSelectionChanged)

		self.timer = QtCore.QTimer(editor)
		self.timer.setInterval(100)
		self.timer.timeout.connect(self.updateHelp)

		cdir = os.path.dirname(os.path.abspath(__file__))

		self.ff = ET.parse(cdir + "/../doc/fileformat.xml")

	def linkClicked(self, url):

		url = url.toString().split("#")
		c = self.editor.textCursor()
		txt = self.editor.toPlainText()

		pos = c.position()
		
		# determine line indent
		txt = txt.replace("\r", "\n").replace("\n\n", "\n,")
		max_indent = 0
		indent = ""
		p = pos
		for i in range(3):
			p = txt.find("\n", p, len(txt))+1
			indent_chars = (len(txt[p:]) - len(txt[p:].lstrip()))
			if indent_chars > max_indent:
				indent = txt[p:(p+indent_chars)]
				max_indent = indent_chars
				break
		p = pos
		for i in range(3):
			p = txt.rfind("\n", 0, p)+1
			indent_chars = (len(txt[p:]) - len(txt[p:].lstrip()))
			if indent_chars > max_indent:
				indent = txt[p:(p+indent_chars)]
				break
			p = p - 2
			if p < 0:
				break

		p = txt.rfind("\n", 0, pos)+1

		if p == pos:
			# at the beginning of a line
			pass
		elif len(txt[p:pos].lstrip()) == 0:
			# already indented
			indent = ""
		else:
			# start new line
			indent = "\n" + indent

		if url[1] == "help":
			self.updateHelpPath([(p, None) for p in url[2:]])
			return
		elif url[1] == "add":
			if url[3] == "empty":
				c.insertText(indent + "<" + url[2] + " />")
				c.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.MoveAnchor, 3)
			else:
				c.insertText(indent + "<" + url[2] + ">" + url[4] + "</" + url[2] + ">")
				c.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.MoveAnchor, len(url[2]) + 3)
				c.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.KeepAnchor, len(url[4]))
				
		elif url[1] == "set":
			pos1 = int(url[5])
			mov = 1
			if txt[pos1-2] == "/":
				c.setPosition(pos1-2)
			else:
				c.setPosition(pos1-1)
			ins = url[2] + '="' + url[3] + '"'
			if (txt[c.position()-1].strip() != ""):
				ins = " " + ins
			if (txt[c.position()].strip() != ""):
				ins += " "
				mov += 1
			c.insertText(ins)
			c.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.MoveAnchor, mov)
			c.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.KeepAnchor, len(url[3]))
		elif url[1] == "ins":
			ins = url[2]
			pos1 = int(url[4])
			pos2 = txt.find("<", pos1)
			if (pos2 >= 0):
				c.setPosition(pos1)
				c.movePosition(QtGui.QTextCursor.Right, QtGui.QTextCursor.KeepAnchor, pos2-pos1)
			c.insertText(ins)
			c.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.KeepAnchor, len(ins))

		self.editor.setTextCursor(c)
		self.editor.setFocus()

	def editorSelectionChanged(self):

		self.timer.start()

	def updateHelp(self):

		self.timer.stop()

		c = self.editor.textCursor()
		pos = c.position()
		txt = self.editor.toPlainText()

		p = re.compile('</?\w+((\s+\w+(\s*=\s*(?:".*?"|\'.*?\'|[\^\'">\s]+))?)+\s*|\s*)/?>')

		items = []
		a = []
		for m in p.finditer(txt):

			if pos < m.start():
				break

			a.append(m)

		items = []
		inside = False
		for i,m in enumerate(a):
			
			inside = (pos >= m.start()) and (pos < m.end())
			closing = (m.group()[0:2] == "</")
			self_closing = (m.group()[-2:] == "/>")
			item = re.search("[a-zA-z0-9_]+", m.group())
			item = item.group()
			is_last = (i == len(a)-1)

			if self_closing and not inside:
				continue

			if len(items) and items[-1][0] == item:
				if not inside:
					items.pop(-1)
				continue

			items.append((item, m))

		self.updateHelpPath(items, inside)

	def getCursorHelp(self):

		c = self.editor.textCursor()
		pos = c.position()
		txt = self.editor.toPlainText()

		for m in re.finditer(r'\b\w+\b', txt):
			if pos >= m.start() and pos <= (m.start() + len(m.group(0))):
				word = m.group(0)
				if word == 'fg':
					word = 'fibergen'
				for k in ["fibergen.%s" % word, "fibergen.FG.%s" % word, word]:
					try:
						#helpstr = pydoc.render_doc(k, "Help on %s", renderer=pydoc.plaintext)
						#helpstr = '<pre>' + html_escape(helpstr) + '</pre>'
						helpstr = pydoc.render_doc(k, "Help on %s", renderer=pydoc.html)
						helpstr = helpstr.replace('&nbsp;', ' ')
						return helpstr
					except:
						pass
				break

		helpstr = "Unknown element"
		return helpstr

	def updateHelpPath(self, items, inside=False):

		#if len(items):
		#	self.scrollToAnchor(items[-1])

		e = self.ff.getroot()
		en = None

		if len(items) and e.tag == items[0][0]:
			en = e
			for item in items[1:]:
				if item[0] == "attrib":
					en = None
					break
				en = e.find(item[0])
				if en is None:
					# try to recover
					en = e.find("actions")
					if en is None:
						break
				e = en

		typ = e.get("type")
		values = e.get("values")

		html = defaultCSS() + """
<style>
h2 {
	margin-top: 0;
}
p {
	margin-top: 1em;
}
p ~ p {
	margin-top: 0;
}
</style>
"""

		html += "<h2>"
		for i, item in enumerate(items):
			if i > 0:
				html += "."
			path = [items[j][0] for j in range(i+1)]
			if i < len(items)-1:
				html += '<a href="http://x#help#' + '#'.join(path) + '">' + item[0] + '</a>'
			else:
				html += item[0]
		html += "</h2>"

		def help_link(tag):
			apath = path + [tag]
			return '<a href="http://x#help#' + '#'.join(apath) + '">' + tag + "</a>"
		
		if en is None:
			helpstr = self.getCursorHelp()
		else:
			helpstr = html_escape(e.get("help"))

		html += '<div class="help">' + helpstr + "</div>"

		if en is None:
			pass
		elif inside or typ != "list":

			if typ != "none":
				html += "<p><b>Type:</b> " + typ + "</p>"

			if typ == "bool":
				values = "0,1"

			if not values is None:
				values = values.split(",")
				values = sorted(values, key=lambda s: s.lower())
				html += "<p><b>Valid values:</b> "
				for i, v in enumerate(values):
					if i > 0:
						html += " | "
					if not item[1] is None:
						html += '<a href="http://x#ins#' + v + '#' + str(item[1].start()) + '#' + str(item[1].end()) + '">' + html_escape(v) + '</a>'
					else:
						html += v
				html += "</p>"

			if not e.text is None and len(e.text.strip()) > 0:
				html += '<p><b>Default:</b> ' + html_escape(e.text.strip()) + "</p>"
			
			if (not en is None):
				attr = ""
				attribs = list(e.findall("attrib"))
				attribs = sorted(attribs, key=lambda a: a.get("name").lower())
				for a in attribs:
					default = html_escape("" if a.text is None else a.text.strip())
					attr += "<tr>"
					if not item[1] is None:
						attr += '<td><b><a href="http://x#set#' + a.get("name") + '#' + default + '#' + str(item[1].start()) + '#' + str(item[1].end()) + '">' + a.get("name") + "</a></b></td>"
					else:
						#attr += '<td><b>' + help_link(a.get("name")) + '</b></td>'
						attr += '<td><b>' + a.get("name") + "</b></td>"
					attr += "<td>" + a.get("type") + "</td>"
					attr += "<td>" + default + "</td>"
					helpstr = a.get("help")
					if not helpstr is None:
						values = a.get("values")
						if not values is None:
							helpstr += " (%s)" % html_escape(values)
						attr += "<td>" + helpstr + "</td>"
					attr += "</tr>"
				if attr != "":
					html += "<h3>Available attributes:</h3>"
					html += '<table>'
					html += "<tr>"
					html += "<th>Name</th>"
					html += "<th>Type</th>"
					html += "<th>Default</th>"
					html += "<th>Description</th>"
					html += "</tr>"
					html += attr
					html += "</table>"
		else:
			tags = ""
			items = list(e.findall("./*"))
			items = sorted(items, key=lambda e: e.tag.lower())
			for a in items:
				if a.tag == "attrib":
					continue
				typ = a.get("type")
				default = html_escape("" if a.text is None else a.text.strip())
				tags += "<tr>"
				if not item[1] is None:
					tags += '<td><b><a href="http://x#add#' + a.tag + '#' + typ + '#' + default + '">' + a.tag + "</a></b></td>"
				else:
					tags += '<td><b>' + help_link(a.tag) + '</b></td>'
				tags += "<td>" + typ + "</td>"
				tags += "<td>" + default + "</td>"
				helpstr = html_escape(a.get("help"))
				helpstr = re.sub('\[(.*?)\]', lambda m: help_link(m.group(1)), helpstr)
				tags += "<td>" + helpstr + "</td>"

				tags += "</tr>"
			if tags != "":
				html += "<h3>Available elements:</h3>"
				html += '<table>'
				html += "<tr>"
				html += "<th>Name</th>"
				html += "<th>Type</th>"
				html += "<th>Default</th>"
				html += "<th>Description</th>"
				html += "</tr>"
				html += tags
				html += "</table>"

		self.updateHtml.emit(html)


class SimpleHelpWidget(QtWidgets.QTextBrowser):

	def __init__(self, editor, parent = None):

		QtWidgets.QTextBrowser.__init__(self, parent)

		self.editor = editor

		self.hwc = HelpWidgetCommon(editor)
		self.hwc.updateHtml.connect(self.updateHtml)

		self.setOpenLinks(False)
		self.anchorClicked.connect(self.hwc.linkClicked)

	def updateHtml(self, html):
		self.setHtml(html)
		self.editor.setFocus()


class HelpWidget(QtWebKitWidgets.QWebView):

	def __init__(self, editor, parent = None):

		QtWebKitWidgets.QWebView.__init__(self, parent)

		self.editor = editor

		self.hwc = HelpWidgetCommon(editor)

		self.mypage = MyWebPage()
		self.mypage.linkClicked.connect(self.hwc.linkClicked)
		self.setPage(self.mypage)

		self.hwc.updateHtml.connect(self.updateHtml)
		
		#self.setStyleSheet("background:transparent");
		#self.setAttribute(QtCore.Qt.WA_TranslucentBackground);

	def updateHtml(self, html):
		self.mypage.setHtml(html)
		self.editor.setFocus()


class DocWidgetCommon(QtCore.QObject):

	updateHtml = QtCore.pyqtSignal('QString')

	def __init__(self, parent = None):

		super(QtCore.QObject, self).__init__(parent)

		cdir = os.path.dirname(os.path.abspath(__file__))

		self.docfile = None

		docfiles = ["../doc/doxygen/html/index.html"] #, "../doc/manual.html"]

		for f in docfiles:
			f = os.path.abspath(os.path.join(cdir, f))
			if os.path.isfile(f):
				self.docfile = f
				break

		if self.docfile is None:
			print("WARNING: No doxygen documentation found! Using online README instead.")
			self.openurl = "https://fospald.github.io/fibergen/"
		else:
			self.openurl = "file://" + self.docfile


class SimpleDocWidget(QtWidgets.QTextBrowser):

	def __init__(self, parent = None):

		QtWidgets.QTextBrowser.__init__(self, parent)

		self.dwc = DocWidgetCommon(self)
		self.setSource(QtCore.QUrl(self.dwc.openurl))


class DocWidget(QtWebKitWidgets.QWebView):

	def __init__(self, parent = None):

		QtWebKitWidgets.QWebView.__init__(self, parent)

		self.dwc = DocWidgetCommon(self)

		self.mypage = MyWebPage()
		self.mypage.setUrl(QtCore.QUrl(self.dwc.openurl))
		self.mypage.linkClicked.connect(self.linkClicked)
		self.setPage(self.mypage)

		css = """
body {
	background-color: white;
}
"""
		data = base64.b64encode(css.encode('utf8')).decode('ascii')
		#self.settings().setUserStyleSheetUrl(QtCore.QUrl("data:text/css;charset=utf-8;base64," + data))

	def linkClicked(self, url):
		self.mypage.setUrl(url)


class DemoWidgetCommon(QtCore.QObject):

	updateHtml = QtCore.pyqtSignal('QString')
	openProjectRequest = QtCore.pyqtSignal('QString')
	newProjectRequest = QtCore.pyqtSignal('QString')

	def __init__(self, parent):

		super(QtCore.QObject, self).__init__(parent)

		cdir = os.path.dirname(os.path.abspath(__file__))
		self.demodir = os.path.abspath(os.path.join(cdir, "../demo"))
		self.simple = False

		self.loadDir()

	def linkClicked(self, url):

		url = url.toString().split("#")
		action = url[1]
		path = os.path.abspath(url[2])

		if action == "cd":
			self.loadDir(path)
		elif action == "open":
			self.openProjectRequest.emit(path)
		elif action == "new":
			self.newProjectRequest.emit(path)


	def loadDir(self, path=None):

		if path is None:
			path = self.demodir

		app = QtWidgets.QApplication.instance()
		pal = app.palette()

		html = defaultCSS() + """
<style>
.demo, .category, .back {
	border: 2px solid """ + pal.link().color().name() + """;
	border-radius: 1em;
	background-color: """ + pal.window().color().name() + """;
	color: """ + pal.buttonText().color().name() + """;
	display: inline-block;
	vertical-align: text-top;
	text-align: center;
	padding: 1em;
	margin: 0.5em;
}
.back {
	font-size: 125%;
	padding: 0.5em;
	margin: 0;
	margin-bottom: 0.5em;
	border-radius: 0.5em;
}
h2 {
	margin-top: 0;
}
.demo:hover, .category:hover, .back:hover {
	border-color: """ + pal.link().color().lighter().name() + """;
}
.demo p {
	margin: 0;
	margin-top: 1em;
	width: 20em;
}
img {
	width: 20em;
	background-color: """ + ("#fff" if self.simple else "initial") + """;
}
.header td {
	border: none;
}
.header td:last-child {
	text-align: right;
}
.header td:first-child {
	white-space: nowrap;
	width: 1%;
}
.header img {
	width: auto;
	height: 5.5em;
	margin-top: -0.5em;
	margin-bottom: -0.5em;
}
.header {
	background-color: """ + ("auto" if self.simple else "initial") + """;
	padding: 1em;
	border-bottom: 1px solid """ + pal.shadow().color().name() + """;
	margin-bottom: 1em;
}
</style>
"""

		category_file = os.path.join(path, "category.xml")

		try:
			if os.path.isfile(category_file):
				xml = ET.parse(category_file).getroot()
			else:
				xml = ET.Element("dummy")
			html += '<table class="header">'
			html += '<tr>'
			if path == self.demodir:
				html += '<td>'
				html += '<h1>' + app.applicationName() + '</h1>'
				html += '<p>A FFT-based homogenization tool.</p>'
				html += '</td>'
				img = xml.find("image")
				if not img is None and not img.text is None and len(img.text) and not self.simple:
					img = os.path.join(path, img.text)
					html += '<td><img src="file://' + img + '" /></td>'
			else:
				html += '<td>'
				title = xml.find("title")
				if not title is None and len(title.text):
					html += '<h1>' + title.text + '</h1>'
				else:
					html += '<h1>' + os.path.basename(path) + '</h1>'
				html += '</td>'
				html += '<td><a class="back" href="http://x#cd#' + path + '/..">&#x21a9; Back</a></td>'
			html += '</tr>'
			html += '</table>'
		except:
			print("error in file", category_file)
			print(traceback.format_exc())
			

		html += '<center class="body">'

		items = []
		indices = []
		dirs = sorted(os.listdir(path), key=lambda s: s.lower(), reverse=True)
		
		img_tag = '<img '
		if self.simple:
			img_tag = '<br/><img width="256" height="256" '

		for d in dirs:

			subdir = os.path.join(path, d)
			if not os.path.isdir(subdir):
				continue

			project_file_xml = os.path.join(subdir, "project.xml")
			project_file_python = os.path.join(subdir, "project.py")
			category_file = os.path.join(subdir, "category.xml")

			item = "<hr/>" if self.simple else ""
			index = None
			if os.path.isfile(project_file_python):
				with open(project_file_python, "rt") as f:
					code = f.read()
				match = re.search("\s*#\s*title\s*:\s*(.*)\s*", code)
				if match:
					title = match.group(1)
				else:
					title = d
				action = "open"
				item += '<a class="demo" href="http://x#' + action + '#' + project_file_python + '">'
				item += '<h2>' + title + '</h2>'
				item += img_tag + ' src="file://' + subdir + '/../category.svg" />'
				match = re.search("\s*#\s*description\s*:\s*(.*)\s*", code)
				if match:
					item += '<p>' + match.group(1) + '</p>'
				item += '</a>'
				index = xml.find("index")
			elif os.path.isfile(project_file_xml):
				try:
					xml = ET.parse(project_file_xml).getroot()
				except:
					print("error in file", project_file_xml)
					print(traceback.format_exc())
					continue
				try:
					action = xml.find("action").text
				except:
					action = "new" if d == "empty" else "open"
				item += '<a class="demo" href="http://x#' + action + '#' + project_file_xml + '">'
				title = xml.find("title")
				if not title is None and not title.text is None and len(title.text):
					item += '<h2>' + title.text + '</h2>'
				else:
					item += '<h2>' + d + '</h2>'
				img = xml.find("image")
				if not img is None and not img.text is None and len(img.text):
					img = os.path.join(subdir, img.text)
					item += img_tag + ' src="file://' + img + '" />'
				else:
					for ext in ["svg", "png"]:
						img = os.path.join(subdir, "thumbnail." + ext)
						if os.path.isfile(img):
							item += img_tag + ' src="file://' + img + '" />'
							break
				desc = xml.find("description")
				if not desc is None and not desc.text is None and len(desc.text):
					item += '<p>' + desc.text + '</p>'
				item += '</a>'
				index = xml.find("index")
			else:
				try:
					if os.path.isfile(category_file):
						xml = ET.parse(category_file).getroot()
					else:
						xml = ET.Element("dummy")
				except:
					print("error in file", category_file)
					print(traceback.format_exc())
					continue
				item += '<a class="category" href="http://x#cd#' + subdir + '">'
				title = xml.find("title")
				if not title is None and not title.text is None and len(title.text):
					item += '<h2>' + title.text + '</h2>'
				else:
					item += '<h2>' + d + '</h2>'
				img = xml.find("image")
				if not img is None and not img.text is None and len(img.text):
					img = os.path.join(subdir, img.text)
					item += img_tag + ' src="file://' + img + '" />'
				item += '</a>'
				index = xml.find("index")

			try:
				index = int(index.text)
			except:
				index = -1

			k = 0
			for k, i in enumerate(indices):
				if i >= index:
					k -= 1
					break
			indices.insert(k+1, index)
			items.insert(k+1, item)

		html += "\n".join(items)


		html += '</center>'

		self.updateHtml.emit(html)


class SimpleDemoWidget(QtWidgets.QTextBrowser):

	openProjectRequest = QtCore.pyqtSignal('QString')
	newProjectRequest = QtCore.pyqtSignal('QString')

	def __init__(self, parent = None):

		QtWidgets.QTextBrowser.__init__(self, parent)

		self.dwc = DemoWidgetCommon(self)
		self.dwc.simple = True
		self.dwc.updateHtml.connect(self.setHtml)
		self.dwc.openProjectRequest.connect(self.emitOpenProjectRequest)
		self.dwc.newProjectRequest.connect(self.emitNewProjectRequest)

		self.setOpenLinks(False)
		self.anchorClicked.connect(self.dwc.linkClicked)

		self.dwc.loadDir()

	def emitOpenProjectRequest(self, path):
		self.openProjectRequest.emit(path)

	def emitNewProjectRequest(self, path):
		self.newProjectRequest.emit(path)


class DemoWidget(QtWebKitWidgets.QWebView):

	openProjectRequest = QtCore.pyqtSignal('QString')
	newProjectRequest = QtCore.pyqtSignal('QString')

	def __init__(self, parent = None):

		QtWebKitWidgets.QWebView.__init__(self, parent)

		self.dwc = DemoWidgetCommon(self)

		self.mypage = MyWebPage()
		self.mypage.linkClicked.connect(self.dwc.linkClicked)
		self.setPage(self.mypage)

		self.dwc.updateHtml.connect(self.mypage.setHtml)
		self.dwc.openProjectRequest.connect(self.emitOpenProjectRequest)
		self.dwc.newProjectRequest.connect(self.emitNewProjectRequest)

		self.dwc.loadDir()

	def emitOpenProjectRequest(self, path):
		self.openProjectRequest.emit(path)

	def emitNewProjectRequest(self, path):
		self.newProjectRequest.emit(path)


class TabDoubleClickEventFilter(QtCore.QObject):

	def eventFilter(self, obj, event):

		if event.type() == QtCore.QEvent.MouseButtonDblClick:
			i = obj.tabAt(event.pos())
			if i >= 0:
				#tab = self.widget(i);
				flags = QtCore.Qt.WindowFlags(QtCore.Qt.Dialog+QtCore.Qt.WindowTitleHint)
				text, ok = QtWidgets.QInputDialog.getText(obj, "Modify tab title", "Please enter new tab title:", text=obj.parent().tabText(i), flags=flags)
				if ok:
					obj.parent().setTabText(i, text)
				return True

		return False


class MainWindow(QtWidgets.QMainWindow):

	def __init__(self, parent = None):
		
		app = QtWidgets.QApplication.instance()
		pal = app.palette()

		QtWidgets.QMainWindow.__init__(self, parent)

		#self.setMinimumSize(1000, 800)
		dir_path = os.path.dirname(os.path.realpath(__file__))
		self.setWindowTitle(app.applicationName() + " - FFT Homogenization Tool")
		self.setWindowIcon(QtGui.QIcon(dir_path + "/../gui/icons/logo1/icon32.png"))


		self.textEdit = XMLTextEdit()

		self.runCount = 0
		self.lastSaveText = self.getSaveText()

		if app.pargs.disable_browser:
			self.helpWidget = SimpleHelpWidget(self.textEdit)
		else:
			self.helpWidget = HelpWidget(self.textEdit)
		vbox = QtWidgets.QVBoxLayout()
		vbox.setContentsMargins(0,0,0,0)
		vbox.addWidget(self.helpWidget)
		helpwrap = QtWidgets.QFrame()
		helpwrap.setLayout(vbox)
		helpwrap.setFrameShape(QtWidgets.QFrame.StyledPanel)
		helpwrap.setFrameShadow(QtWidgets.QFrame.Sunken)
		helpwrap.setStyleSheet("background-color:%s;" % pal.base().color().name());

		self.tabWidget = QtWidgets.QTabWidget()
		self.tabWidget.setTabsClosable(True)
		self.tabWidget.setMovable(True)
		self.tabWidget.tabCloseRequested.connect(self.tabCloseRequested)
		self.tabWidget.tabBar().installEventFilter(TabDoubleClickEventFilter(self.tabWidget))
		#self.tabWidget.tabBar().setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
		#self.tabWidget.tabBar().setExpanding(True)

		self.filename = None
		self.filetype = None
		self.file_id = 0
		self.filenameLabel = QtWidgets.QLabel()

		self.statusBar = QtWidgets.QStatusBar()
		self.statusBar.showMessage("")
		#self.statusBar.addPermanentWidget(self.filenameLabel)
		self.textEdit.cursorPositionChanged.connect(self.updateStatus)

		self.demoTab = None
		self.demoTabIndex = None
		self.docTab = None
		self.docTabIndex = None

		self.vSplit = QtWidgets.QSplitter(self)
		self.vSplit.setOrientation(QtCore.Qt.Vertical)
		self.vSplit.insertWidget(0, self.textEdit)
		self.vSplit.insertWidget(1, helpwrap)
		#self.vSplit.insertWidget(2, self.statusBar)
		self.setStatusBar(self.statusBar)

		self.hSplit = QtWidgets.QSplitter(self)
		self.hSplit.setOrientation(QtCore.Qt.Horizontal)
		self.hSplit.insertWidget(0, self.vSplit)
		self.hSplit.insertWidget(1, self.tabWidget)

		# search for a good icon theme

		def get_size(start_path = '.'):
			total_size = 0
			for dirpath, dirnames, filenames in os.walk(start_path):
				for f in filenames:
					fp = os.path.join(dirpath, f)
					try:
						total_size += os.path.getsize(fp)
					except:
						pass
			return total_size

		themes = []
		for path in QtGui.QIcon.themeSearchPaths():
			if os.path.isdir(path):
				for name in os.listdir(path):
					dirname = os.path.join(path, name)
					if os.path.isdir(dirname):
						themes.append((name, get_size(dirname)))

		themes = sorted(themes, key=lambda tup: tup[1], reverse=True)

		for theme, size in themes:
			QtGui.QIcon.setThemeName(theme)
			if QtGui.QIcon.hasThemeIcon("document-new"):
				#print("selected theme:", theme)
				break

		# add toolbar actions

		def aa(icon, text, func, key):
			action = self.toolbar.addAction(QtGui.QIcon.fromTheme(icon), text)
			action.triggered.connect(func)
			action.setShortcut(key)
			action.setToolTip("%s (%s)" % (text, str(QtGui.QKeySequence(key).toString())))
			return action

		self.toolbar = QtWidgets.QToolBar()
		self.toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
		self.toolbar.setObjectName("toolbar")

		# https://specifications.freedesktop.org/icon-naming-spec/icon-naming-spec-latest.html
		aa("document-new", "New", self.newProjectGui, QtCore.Qt.CTRL + QtCore.Qt.Key_N)
		aa("document-open", "Open", self.openProjectGui, QtCore.Qt.CTRL + QtCore.Qt.Key_O)
		self.saveSeparator = self.toolbar.addSeparator()
		self.saveAction = aa("document-save", "Save", self.saveProjectGui, QtCore.Qt.CTRL + QtCore.Qt.Key_S)
		self.saveAsAction = aa("document-save-as", "Save As", lambda: self.saveProjectGui(True), QtCore.Qt.CTRL + QtCore.Qt.SHIFT + QtCore.Qt.Key_S)
		self.undoSeparator = self.toolbar.addSeparator()
		self.undoAction = aa("edit-undo", "Undo", self.undo, QtCore.Qt.CTRL + QtCore.Qt.Key_Z)
		self.redoAction = aa("edit-redo", "Redo", self.redo, QtCore.Qt.CTRL + QtCore.Qt.SHIFT + QtCore.Qt.Key_Z)
		self.runSeparator = self.toolbar.addSeparator()
		self.runAction = aa("media-playback-start", "Run", self.runProject, QtCore.Qt.CTRL + QtCore.Qt.Key_R)
		spacer = QtWidgets.QWidget()
		spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
		self.toolbar.addWidget(spacer)
		aa("preferences-system", "Preferences", self.openPreferences, 0)
		self.toolbar.addSeparator()
		aa("help-contents", "Help", self.openHelp, QtCore.Qt.Key_F1)
		aa("help-about", "About", self.openAbout, 0)
		self.toolbar.addSeparator()
		aa("application-exit", "Exit", self.exit, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)

		for a in [self.redoAction, self.undoAction]:
			a.setEnabled(False)

		self.textEdit.undoAvailable.connect(self.undoAvailable)
		self.textEdit.redoAvailable.connect(self.redoAvailable)

		self.addToolBar(self.toolbar)
		self.setCentralWidget(self.hSplit)

		try:
			app.restoreWindowState(self, "main")
			val = app.settings.value("hSplitterSize")
			if not val is None:
				self.hSplit.restoreState(val)
			val = app.settings.value("vSplitterSize")
			if not val is None:
				self.vSplit.restoreState(val)
		except:
			#print(traceback.format_exc())
			screen = app.desktop().screenGeometry()
			w = screen.width()
			h = screen.height()
			self.resize(w*2/3, h*2/3)
			self.setWindowState(QtCore.Qt.WindowMaximized)
			self.hSplit.setSizes([w/3, 2*w/3])
			self.vSplit.setSizes([2*h/3, h/3])

		self.setDocumentVisible(False)
		self.tabWidget.setVisible(False)

		self.show()

	def openPreferences(self):
		w = PreferencesWidget()
		w.exec_()

	def setDocumentVisible(self, visible):
		self.vSplit.setVisible(visible)
		for a in [self.saveSeparator, self.saveAction, self.saveAsAction, self.undoSeparator, self.undoAction, self.redoAction, self.runSeparator, self.runAction]:
			a.setVisible(visible)

	def exit(self):
		self.close()

	def undoAvailable(self, b):
		self.undoAction.setEnabled(b)

	def redoAvailable(self, b):
		self.redoAction.setEnabled(b)

	def undo(self):
		self.textEdit.undo()

	def redo(self):
		self.textEdit.redo()

	def openAbout(self):
		app = QtWidgets.QApplication.instance()
		webbrowser.open('https://fospald.github.io/' + app.applicationName() + '/')

	def openHelp(self):
		if self.docTabIndex is None:
			if self.docTab is None:
				app = QtWidgets.QApplication.instance()
				if app.pargs.disable_browser:
					self.docTab = SimpleDocWidget()
				else:
					self.docTab = DocWidget()
			self.docTabIndex = self.addTab(self.docTab, "Help")
		self.tabWidget.setCurrentWidget(self.docTab)

	def updateStatus(self):
		c = self.textEdit.textCursor()
		pos = c.position()
		base = 1
		self.statusBar.showMessage(
			"  Line: " + str(c.blockNumber()+base) +
			"  Column: " + str(c.columnNumber()+base) +
			"  Char: " + str(c.position()+base) +
			("" if self.filename is None else ("  File: " + self.filename))
		)
		#self.filenameLabel.setText("" if self.filename is None else self.filename)

	def addTab(self, widget, title):
		index = self.tabWidget.addTab(widget, title)
		self.tabWidget.setCurrentIndex(index)
		self.tabWidget.setVisible(True)
		return index

	def tabCloseRequested(self, index):
		if index == self.tabWidget.indexOf(self.demoTab):
			self.demoTabIndex = None
		elif index == self.tabWidget.indexOf(self.docTab):
			self.docTabIndex = None
		self.tabWidget.removeTab(index)
		if self.tabWidget.count() == 0:
			self.tabWidget.setVisible(False)

	def closeEvent(self, event):

		if not self.checkTextSaved():
			event.ignore()
		else:

			app = QtWidgets.QApplication.instance()
			app.saveWindowState(self, "main")
			app.settings.setValue("hSplitterSize", self.hSplit.saveState())
			app.settings.setValue("vSplitterSize", self.vSplit.saveState())
			app.settings.sync()

			event.accept()

	def openProjectGui(self):
		if not self.checkTextSaved():
			return
		filename, _filter = QtWidgets.QFileDialog.getOpenFileName(self, "Open Project", os.getcwd(), "XML Files (*.xml)")
		if (filename != ""):
			self.openProject(filename)
	
	def checkTextSaved(self):
		if self.lastSaveText != self.getSaveText():
			r = QtWidgets.QMessageBox.warning(self, "Warning", "Your text has not been saved! Continue without saving?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
			return r == QtWidgets.QMessageBox.Yes
		return True

	def getSaveText(self):
		return self.textEdit.toPlainText().encode('utf8')

	def saveProjectGui(self, save_as=False):
		if (not self.filename is None and not save_as):
			filename = self.filename
		else:
			filename, _filter = QtWidgets.QFileDialog.getSaveFileName(self, "Save Project", os.getcwd(), "XML Files (*.xml)")
			if (filename == ""):
				return False
		try:
			txt = self.getSaveText()
			with open(filename, "wb+") as f:
				f.write(txt)
			self.lastSaveText = txt
			self.filename = filename
			self.filetype = os.path.splitext(filename)[1]
			self.updateStatus()
			return True
		except:
			QtWidgets.QMessageBox.critical(self, "Error", traceback.format_exc())
			return False
	
	def openDemo(self, filename):
		if not self.openProjectSave(filename):
			return False
		if self.tabWidget.currentWidget() == self.demoTab:
			self.tabCloseRequested(self.tabWidget.currentIndex())
			self.demoTabIndex = None
		return True

	def openProjectSave(self, filename):
		if not self.checkTextSaved():
			return False
		return self.openProject(filename)

	def openProject(self, filename):
		try:
			filename = os.path.realpath(filename)
			filedir = os.path.dirname(filename)
			os.chdir(filedir)
			with open(filename, "rt") as f:
				txt = f.read()
			self.textEdit.setPlainText(txt)
			self.filename = filename
			self.filetype = os.path.splitext(filename)[1]
			self.file_id += 1
			self.lastSaveText = self.getSaveText()
			self.textEdit.document().clearUndoRedoStacks()
			self.setDocumentVisible(True)
			self.updateStatus()
		except:
			QtWidgets.QMessageBox.critical(self, "Error", traceback.format_exc())
			return False
		return True

	def newProjectGui(self):
		if self.demoTabIndex is None:
			if self.demoTab is None:
				app = QtWidgets.QApplication.instance()
				if app.pargs.disable_browser:
					self.demoTab = SimpleDemoWidget()
				else:
					self.demoTab = DemoWidget()
				self.demoTab.openProjectRequest.connect(self.openDemo)
				self.demoTab.newProjectRequest.connect(self.newProject)
			self.demoTabIndex = self.addTab(self.demoTab, "Demos")
		self.tabWidget.setCurrentWidget(self.demoTab)

	def newProject(self, filename=""):
		if not self.checkTextSaved():
			return False
		if self.tabWidget.currentWidget() == self.demoTab:
			self.tabCloseRequested(self.tabWidget.currentIndex())
			self.demoTabIndex = None
		txt = ""
		try:
			with open(filename, "rt") as f:
				txt = f.read()
		except:
			pass
		self.textEdit.setPlainText(txt)
		self.filename = None
		self.filetype = os.path.splitext(filename)[1]
		self.file_id += 1
		self.lastSaveText = self.getSaveText()
		self.textEdit.document().clearUndoRedoStacks()
		self.setDocumentVisible(True)
		self.updateStatus()
		return True

	def runProject(self, fg=None):

		if self.filetype == ".py":
			# run pure python code
			py = str(self.textEdit.toPlainText())
			loc = dict()
			glob = dict()
			exec(py, glob, loc)
			return
		
		app = QtWidgets.QApplication.instance()

		if not isinstance(fg, fibergen.FG):
			try:
				fg = fibergen.FG()
				fg.set_py_enabled(not app.pargs.disable_python)
				xml = str(self.textEdit.toPlainText())
				fg.set_xml(xml)
			except:
				print(traceback.format_exc())
		else:
			xml = fg.get_xml()
			self.textEdit.setPlainText(xml)
		
		try:
			xml_root = ET.fromstring(xml)
		except:
			xml_root = None
			print(traceback.format_exc())

		coord_names1 = ["x", "y", "z"]
		coord_names2 = ["xx", "yy", "zz", "yz", "xz", "xy", "zy", "zx", "yx"]

		field_labels = {"phi": lambda i: (u"φ_%d" % i, "phase %d" % i),
			"epsilon": lambda i: (u"ε_%s" % coord_names2[i], "strain %s" % coord_names2[i]),
			"sigma": lambda i: (u"σ_%s" % coord_names2[i], "stress %s" % coord_names2[i]),
			"u": lambda i: ("u_%s" % coord_names1[i], "displacement %s" % coord_names1[i]),
			"normals": lambda i: ("n_%s" % coord_names1[i], "normal %s" % coord_names1[i]),
			"orientation": lambda i: ("o_%s" % coord_names1[i], "orientation %s" % coord_names1[i]),
			"fiber_translation": lambda i: ("t_%s" % coord_names1[i], "fiber_translation %s" % coord_names1[i]),
			"fiber_id": lambda i: ("fid", "fiber id"),
			"p": lambda i: ("p", "pressure"),
			"distance": lambda i: ("d", "normal distance"),
			"material_id": lambda i: ("mid", "material id"),
		}

		extra_fields_list = ["fiber_id", "fiber_translation", "normals", "orientation", "distance"]
		extra_fields = []
		record_loadstep = -1

		if not xml_root is None:
			view = xml_root.find("view")
			if not view is None:
				rl = view.find("record_loadstep")
				if not rl is None:
					record_loadstep = int(rl.text)
				ef = view.find("extra_fields")
				if not ef is None:
					ef = ef.text.split(",")
					for f in ef:
						if f in extra_fields_list:
							extra_fields.append(f)

		phase_fields = ["material_id", "phi"]
		run_fields = ["epsilon", "sigma", "u"]
		const_fields = phase_fields + extra_fields
		field_names = phase_fields + run_fields + extra_fields

		try:
			mode = fg.get("solver.mode".encode('utf8'))
		except:
			mode = "elasticity"

		if (mode == "viscosity"):
			field_labels["epsilon"] = lambda i: (u"σ_%s" % coord_names2[i], "fluid stress %s" % coord_names2[i])
			field_labels["sigma"] = lambda i: (u"ɣ_%s" % coord_names2[i], "shear rate %s" % coord_names2[i])
			run_fields.append("p")

		if (mode == "heat"):
			field_labels["u"] = lambda i: ("T", "temperature")
			field_labels["epsilon"] = lambda i: (u"∇T_%s" % coord_names1[i], "temperature gradient %s" % coord_names1[i])
			field_labels["sigma"] = lambda i: ("q_%s" % coord_names1[i], "heat flux %s" % coord_names1[i])
		if (mode == "porous"):
			field_labels["u"] = lambda i: ("p", "pressure")
			field_labels["epsilon"] = lambda i: (u"∇p_%s" % coord_names1[i], "pressure gradient %s" % coord_names1[i])
			field_labels["sigma"] = lambda i: ("v_%s" % coord_names1[i], "volumetric flux %s" % coord_names1[i])

		field_groups = []
		mean_strains = []
		mean_stresses = []
		loadstep_called = []
		phase_names = []

		get_mean_values = False

		progress = QtWidgets.QProgressDialog("Computation is running...", "Cancel", 0, 0, self)
		progress.setWindowTitle("Run")
		progress.setWindowFlags(progress.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

		#progress.setWindowModality(QtCore.Qt.WindowModal)
		#tol = fg.get("solver.tol".encode('utf8'))

		def process_events():
			for i in range(5):
				QtWidgets.QApplication.processEvents()

		def loadstep_callback():
			
			process_events()

			loadstep_called.append(1)

			if record_loadstep >= 0 and record_loadstep != len(loadstep_called):
				return progress.wasCanceled()

			if len(field_groups) == 0:
				phase_names = fg.get_phase_names()
				field_labels['phi'] = lambda i: (phase_names[i], "%s material" % phase_names[i])
				phi = fg.get_field("phi")
				for name in field_names:
					num_discrete_values = float('inf')
					value_labels = {}
					if name == "material_id":
						ids = np.array(range(phi.shape[0]))
						#data = np.expand_dims(np.argmax(ids[:,np.newaxis,np.newaxis,np.newaxis]*phi, axis=0), axis=0)
						data = np.expand_dims(np.argmax(phi, axis=0), axis=0)
						num_discrete_values = len(ids)
						value_labels = {i+0.5: field_labels['phi'](i)[0] for i in ids}
					elif name == "phi":
						data = phi
					else:
						data = fg.get_field(name.encode('utf8'))
					shape = data.shape
					fields = []
					#print(name, shape)
					for i in range(shape[0]):
						field = PlotField()
						field.data = [data]
						field.label, field.description = field_labels[name](i)
						field.name = name
						field.key = name if shape[0] == 1 else (name + str(i))
						field.component = i
						field.num_discrete_values = num_discrete_values
						field.value_labels = value_labels
						#field.amin = np.amin(data[i])
						#field.amax = np.amax(data[i])
						fields.append(field)
					field_groups.append(fields)
					process_events()
			else:
				for field_group in field_groups:
					for field in field_group:
						if field.name in const_fields:
							data = field.data[-1]
						else:
							data = fg.get_field(field.name.encode('utf8'))
						field.data.append(data)
						process_events()

			return progress.wasCanceled()


		def convergence_callback():
			#residual = fg.get_residuals()[-1]
			#print "res=", residual
			# progress.setValue(100)
			process_events()
			if get_mean_values:
				mean_strains.append(fg.get_mean_strain())
				process_events()
				mean_stresses.append(fg.get_mean_stress())
				process_events()
			return progress.wasCanceled()


		try:
			fg.set_loadstep_callback(loadstep_callback)
			fg.set_convergence_callback(convergence_callback)
			#print fg.get_xml()

			progress.show()
			process_events()
			
			print("Running FG with id", id(fg))
			fg.run()

			if progress.wasCanceled():
				progress.close()
				del fg
				return

			if record_loadstep != 0:
				fg.init_phase()

			if len(loadstep_called) == 0 and record_loadstep != 0:
				field_names = const_fields
				loadstep_callback()

		except:
			print(traceback.format_exc())

		progress.close()
		process_events()
		
		self.runCount += 1

		if len(field_groups) == 0:
			del fg
			return

		# compute magnitudes
		for i, field_group in enumerate(field_groups):
			name = field_group[0].name
			dim = field_group[0].data[0].shape[0]
			if name in ["u", "sigma", "epsilon"] and dim > 1:
				mag_name = "magnitude_" + name
				field = PlotField()
				if dim == 6:
					field.data = [(np.linalg.norm(d[0:3], axis=0, keepdims=True)**2 + 2*(np.linalg.norm(d[3:6], axis=0, keepdims=True)**2))**0.5 for d in field_group[0].data]
				else:
					field.data = [np.linalg.norm(d, axis=0, keepdims=True) for d in field_group[0].data]
				field.label = "‖" + field_group[0].label[0:field_group[0].label.find("_")] + "‖"
				field.description = "magnitude of " + field_group[0].description[0:(field_group[0].label.rfind(" ")-1)]
				field.name = mag_name
				field.key = mag_name
				field.component = 0
				field.num_discrete_values = field_group[0].num_discrete_values
				field.value_labels = field_group[0].value_labels
				field_groups[i].append(field)
				#field_groups[i].insert(0, field)


		volume_fractions = collections.OrderedDict()
		phase_names = fg.get_phase_names()
		for key in phase_names:
			volume_fractions[key] = fg.get_volume_fraction(key)

		def section(text):
			return "<h2>%s</h2>\n" % text

		def matrix(a):
			if isinstance(a, np.ndarray):
				a = a.tolist()
			if isinstance(a, float):
				return "%.04g" % a
			if not isinstance(a, list):
				return str(a)
			tab = "<table>\n"
			for r in a:
				tab += "<tr>\n"
				if isinstance(r, collections.abc.Iterable):
					for c in r:
						tab += "<td>%s</td>\n" % matrix(c)
				else:
					tab += "<td>%s</td>\n" % matrix(r)
				tab += "</tr>\n"
			tab += "</table>\n"
			return tab
					
		def table(a):
			tab = "<table>\n"
			if isinstance(a, dict):
				for key, value in a.items():
					if isinstance(value, dict):
						tab += "<tr>\n"
						tab += "<th></th>\n"
						for th_key in value.keys():
							tab += "<th>%s</th>\n" % th_key
						tab += "</tr>\n"
					break
				for key, value in a.items():
					tab += "<tr>\n"
					tab += "<td>%s</td>\n" % key
					if isinstance(value, dict):
						for k, v in value.items():
							tab += "<td>%s</td>\n" % matrix(v)
					else:
						tab += "<td>%s</td>\n" % matrix(value)
					tab += "</tr>\n"
			elif isinstance(a, list):
				for value in a:
					if isinstance(value, dict):
						tab += "<tr>\n"
						for th_key in value.keys():
							tab += "<th>%s</th>\n" % th_key
						tab += "</tr>\n"
					break
				for value in a:
					if isinstance(value, dict):
						tab += "<tr>\n"
						for k, v in value.items():
							tab += "<td>%s</td>\n" % matrix(v)
						tab += "</tr>\n"
				
			else:
				return str(a)
			tab += "</table>\n"
			return tab

		def plot(x, y, title, xlabel, ylabel, yscale="linear"):
			fig, ax = plt.subplots(nrows=1, ncols=1)
			ax.plot(x, y, 'ro-')
			plt.grid()
			ax.set_title(title)
			ax.set_xlabel(xlabel)
			ax.set_ylabel(ylabel)
			ax.set_yscale(yscale)
			tf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
			fig.savefig(tf.name, transparent=True)
			plt.close(fig)	# close the figure
			img = "<p><img class=\"plot\" src='file://%s' /></p>" % tf.name
			img += "<p>%s</p>" % tf.name
			return img

		resultText = ""

		resultText += section('General')
		resultText += table(collections.OrderedDict([
			('solve_time', fg.get_solve_time()),
			('error', fg.get_error()),
			('distance_evals', fg.get_distance_evals()),
		]))
		
		residuals = fg.get_residuals()
		if len(residuals) > 0:
			resultText += section('Residual plot')
			resultText += plot(range(len(residuals)), residuals, "Residuals", "Iteration", "Residual", "log")

		resultText += section('Volume fractions')
		resultText += table(volume_fractions)

		resultText += section('FO tensors')
		resultText += table(collections.OrderedDict([
			('A2', matrix(fg.get_A2())),
			('A4', matrix(fg.get_A4())),
		]))

		def safe_call(func):
			try:
				return func()
			except:
				return None

		def mat(a):
			if (len(a) == 6):
				return np.array([
					[a[0], a[5], a[4]],
					[a[5], a[1], a[3]],
					[a[4], a[3], a[2]]
				], dtype=np.double)
			if (len(a) == 9):
				return np.array([
					[a[0], a[5], a[4]],
					[a[8], a[1], a[3]],
					[a[7], a[6], a[2]]
				], dtype=np.double)
			return a

		resultText += section('Mean quantities')
		resultText += table(collections.OrderedDict([
			('mean_stress', matrix(mat(fg.get_mean_stress()))),
			('mean_strain', matrix(mat(fg.get_mean_strain()))),
			#('mean_cauchy_stress', matrix(mat(fg.get_mean_cauchy_stress()))),
			#('mean_energy', safe_call(fg.get_mean_energy)),
			('effective_property', matrix(fg.get_effective_property())),
		]))

		if len(residuals) > 0:
			resultText += section('Residuals')
			resultText += table(collections.OrderedDict([
				('residuals', matrix([[i,r] for i,r in enumerate(residuals)])),
			]))
		
		if get_mean_values:
			for i, ij in enumerate([11, 22, 33, 23, 13, 12]):
				resultText += plot(range(len(mean_stresses)), [s[i] for s in mean_stresses], "Sigma_%s" % ij, "Iteration", "Sigma_%d" % ij, "linear")
				resultText += plot(range(len(mean_strains)), [s[i] for s in mean_strains], "Epsilon_%s" % ij, "Iteration", "Epsilon_%d" % ij, "linear")

		other = self.tabWidget.currentWidget()

		if not isinstance(other, PlotWidget):
			other = None
		elif other.file_id != self.file_id:
			other = None

		rve_dims = fg.get_rve_dims()

		tab = PlotWidget(rve_dims, field_groups, extra_fields, xml, xml_root, resultText, other)
		tab.file_id = self.file_id

		if len(tab.fields) > 0:
			i = self.addTab(tab, "Run_%d" % self.runCount)

		del fg


class App(QtWidgets.QApplication):

	def __init__(self, args):

		# parse arguments
		parser = argparse.ArgumentParser(description='fibergen - A FFT-based homogenization tool.')
		parser.add_argument('project', metavar='filename', nargs='?', help='xml project filename to load')
		parser.add_argument('--disable-browser', action='store_true', default=(not "QtWebKitWidgets" in globals()), help='disable browser components')
		parser.add_argument('--disable-python', action='store_true', default=False, help='disable Python code evaluation in project files')
		parser.add_argument('--style', default="", help='set application style')
		self.pargs = parser.parse_args(args[1:])
		print(self.pargs)

		QtWidgets.QApplication.__init__(self, list(args) + ["--disable-web-security"])

		self.setApplicationName("fibergen")
		self.setApplicationVersion("2020.1")
		self.setOrganizationName("NumaPDE")

		print("matplotlib:", matplotlib.__version__, "numpy:", np.__version__)

		if self.pargs.style != "":
			styles = QtWidgets.QStyleFactory.keys()
			if not self.pargs.style in styles:
				print("Available styles:", styles)
				raise "unknown style"
			self.setStyle(self.pargs.style)

		# set matplotlib defaults
		font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.GeneralFont)
		mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
		
		pal = self.palette()
		text_color = pal.text().color().name()
		bg_color = pal.base().color().name()
		rcParams.update({
			'figure.autolayout': True,
			'font.size': font.pointSize(),
			'font.family': "monospace",
			'font.monospace': [mono.family()] + rcParams['font.monospace'],
			'font.sans-serif': [font.family()] + rcParams['font.sans-serif'],
			'text.color': text_color,
			'axes.labelcolor': text_color,
			'xtick.color': text_color,
			'ytick.color': text_color,
			#'figure.facecolor': bg_color,
			#'savefig.facecolor': bg_color,
			#'axes.facecolor': bg_color,
			'backend': 'Qt5Agg',
		})
		#print(rcParams)

		self.settings = QtCore.QSettings(self.organizationName(), self.applicationName())
		self.window = MainWindow()

		print("settings:", self.settings.fileName())

		try:
			if (not self.pargs.project is None):
				self.window.openProject(self.pargs.project)
				#self.window.runProject()
			else:
				self.window.newProjectGui()
		except:
			print(traceback.format_exc())

	def notify(self, receiver, event):
		try:
			QtWidgets.QApplication.notify(self, receiver, event)
		except e:
			QtWidgets.QMessageBox.critical(self, "Error", traceback.format_exc())
		return False
	
	def restoreWindowState(self, win, prefix):
		val = self.settings.value(prefix + "_geometry")
		if not val is None:
			win.restoreGeometry(val)
		if (isinstance(win, QtWidgets.QMainWindow)):
			val = self.settings.value(prefix + "_windowState")
			if not val is None:
				win.restoreState(val)

	def saveWindowState(self, win, prefix):
		self.settings.setValue(prefix + "_geometry", win.saveGeometry())
		if (isinstance(win, QtWidgets.QMainWindow)):
			self.settings.setValue(prefix + "_windowState", win.saveState())


# Call this function in your main after creating the QApplication
def setup_interrupt_handling():
	# Setup handling of KeyboardInterrupt (Ctrl-C) for PyQt.
	signal.signal(signal.SIGINT, _interrupt_handler)
	# Regularly run some (any) python code, so the signal handler gets a
	# chance to be executed:
	safe_timer(50, lambda: None)

# Define this as a global function to make sure it is not garbage
# collected when going out of scope:
def _interrupt_handler(signum, frame):
	# Handle KeyboardInterrupt: quit application.
	QtGui.QApplication.quit()
	print("_interrupt_handler")

def safe_timer(timeout, func, *args, **kwargs):
	# Create a timer that is safe against garbage collection and overlapping
	# calls. See: http://ralsina.me/weblog/posts/BB974.html
	def timer_event():
		try:
			func(*args, **kwargs)
		finally:
			QtCore.QTimer.singleShot(timeout, timer_event)
	QtCore.QTimer.singleShot(timeout, timer_event)

def eh():
	print("error")
	traceback.print_exception()

if __name__ == "__main__":
	#sys.excepthook = eh
	app = App(sys.argv)
	#setup_interrupt_handling()
	ret = app.exec_()
	sys.exit(ret)

