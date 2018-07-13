#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, division

import fibergen
import sys, os, re
import webbrowser
import base64
import traceback
import codecs
import collections
import tempfile
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import scipy.misc

from PyQt5 import QtCore, QtGui, QtWidgets
try:
	from PyQt5 import QtWebKitWidgets
except:
	from PyQt5 import QtWebEngineWidgets as QtWebKitWidgets
	QtWebKitWidgets.QWebView = QtWebKitWidgets.QWebEngineView
	QtWebKitWidgets.QWebPage = QtWebKitWidgets.QWebEnginePage

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

	def addStretch(self, stretch):
		pass

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
		spaceX = self.spacing()

		for item in self.itemList:

			if isinstance(item, QtWidgets.QSpacerItem):
				x += item.sizeHint().width()
				continue

			spaceX = self.spacing() #+ wid.style().layoutSpacing(QtWidgets.QSizePolicy.PushButton, QtWidgets.QSizePolicy.PushButton, QtCore.Qt.Horizontal)
			spaceY = self.spacing() #+ wid.style().layoutSpacing(QtWidgets.QSizePolicy.PushButton, QtWidgets.QSizePolicy.PushButton, QtCore.Qt.Vertical)
			nextX = x + item.sizeHint().width()

			if nextX > rect.right():
				x = rect.x()
				y = y + lineHeight + spaceY
				nextX = x + item.sizeHint().width()

			if not testOnly:
				item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), item.sizeHint()))

			x = nextX + spaceX
			lineHeight = max(lineHeight, item.sizeHint().height())

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

	def defaultCSS(self, tags=True):

		app = QtWidgets.QApplication.instance()
		pal = app.palette()
		font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.GeneralFont)

		html = """
a, html, body, div, span, p, h1, h2, h3, tr, td, th {
	font-family: \"""" + font.family() + """\";
	font-size: """ + str(font.pointSize()) + """pt;
}
body {
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
}
.help {
	background-color: """ + pal.toolTipBase().color().name() + """;
	color: """ + pal.toolTipText().color().name() + """;
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

	def __init__(self, field_groups, xml, resultText, other = None, parent = None):

		app = QtWidgets.QApplication.instance()
		pal = app.palette()

		self.changeFieldSuspended = False
		self.replot_reset_limits = False

		QtWidgets.QWidget.__init__(self, parent)
		
		vbox = QtWidgets.QVBoxLayout()
		#vbox.setContentsMargin(0)

		self.fig = Figure(figsize=(20,20))
		self.fig.set_tight_layout(None)
		self.fig.set_frameon(False)
		self.cb = None

		self.axes = self.fig.add_subplot(111)
		self.axes.set_xlabel("x")
		self.axes.set_ylabel("y")

		vbox = QtWidgets.QVBoxLayout(self)

		def makeChangeFieldCallback(index):
			return lambda checked: self.changeField(index, checked)
		
		self.fields = []
		self.currentFieldIndex = other.currentFieldIndex if (other != None) else None

		spacing = 2

		flow = FlowLayout()
		for j, field_group in enumerate(field_groups):
			gbox = QtWidgets.QHBoxLayout()
			for i, field in enumerate(field_group):
				button = QtWidgets.QToolButton()
				field.button = button
				button.setText(field.label)
				button.setCheckable(True)
				index = len(self.fields)
				button.toggled.connect(makeChangeFieldCallback(index))
				if i > 0:
					gbox.addSpacing(spacing)
				gbox.addWidget(button)
				self.fields.append(field)
			if j > 0:
				flow.addSpacing(spacing*4)
			flow.addLayout(gbox)

		hbox = QtWidgets.QHBoxLayout()
		hbox.setAlignment(QtCore.Qt.AlignTop)
		hbox.setSpacing(spacing)
		hbox.addLayout(flow)
		hbox.addSpacing(4*spacing)

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

		vbox.addLayout(hbox)

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
			self.sliceSlider.setValue((self.sliceSlider.maximum() + self.sliceSlider.minimum())/2)
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

		self.colormapCombo = QtWidgets.QComboBox()
		self.colormapCombo.setEditable(False)
		colormaps = sorted(mcmap.datad)
		for cm in colormaps:
			self.colormapCombo.addItem(cm)
		self.colormapCombo.setCurrentIndex(colormaps.index("jet"))
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

		hbox = QtWidgets.QHBoxLayout()
		hbox.addWidget(QtWidgets.QLabel("Colormap:"))
		hbox.addWidget(self.colormapCombo)
		hbox.addWidget(QtWidgets.QLabel("Contrast:"))
		hbox.addWidget(self.alphaSlider)
		hbox.addWidget(self.alphaLabel)
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
		self.replotButton.clicked.connect(self.replot)

		hbox = QtWidgets.QHBoxLayout()
		hbox.addWidget(QtWidgets.QLabel("Bounds:"))
		hbox.addWidget(self.customBoundsCheck)
		hbox.addWidget(self.vminLabel)
		hbox.addWidget(self.vminText)
		hbox.addWidget(self.vmaxLabel)
		hbox.addWidget(self.vmaxText)
		hbox.addWidget(self.replotButton)
		vbox.addLayout(hbox)

		self.stack = QtWidgets.QStackedWidget()
		self.stack.setFrameShape(QtWidgets.QFrame.StyledPanel)
		self.stack.setFrameShadow(QtWidgets.QFrame.Sunken)

		vbox.addWidget(self.stack)
		
		self.figcanvas = FigureCanvas(self.fig)
		self.figcanvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
		#self.figcanvas.setContentsMargins(0, 0, 0, 0)
		#self.figcanvas.setStyle(app.style())

		self.fignavbar = NavigationToolbar(self.figcanvas, self)
		self.fignavbar.set_cursor(cursors.SELECT_REGION)
		self.fignavbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)

		def setIcon(c, names):
			for name in names:
				if QtGui.QIcon.hasThemeIcon(name):
					c.setIcon(QtGui.QIcon.fromTheme(name))
					return True
			return False
				
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

		wvbox = QtWidgets.QVBoxLayout()
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

		if 1:
			self.resultTextEdit = QtWebKitWidgets.QWebView()
			self.resultPage = MyWebPage()
			self.resultPage.setHtml(self.resultPage.defaultCSS() + resultText)
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
	
		self.updateFigCanvasVisible()

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
		self.resetBounds()
		self.replot()
		self.loadstepLabel.setText("%04d" % self.loadstepSlider.value())

	def sliceComboChanged(self, index):
		self.replot_reset_limits = True
		if self.sliceSlider.value() == 0:
			self.sliceSliderChanged()
		else:
			self.sliceSlider.setValue(0)
		data_shape = self.fields[0].data[0].shape[1:4]
		self.sliceSlider.setMaximum(data_shape[index]-1)

	def customBoundsCheckChanged(self, state):
		enable = (state != 0)
		self.vminText.setEnabled(enable)
		self.vmaxText.setEnabled(enable)
		self.resetBounds()
		if (state == 0):
			self.replot()

	def getAlpha(self):
		return 0.4999*(self.alphaSlider.value()/self.alphaSlider.maximum())**3
	
	def getCurrentSlice(self):
		field = self.fields[self.currentFieldIndex]
		s_index = self.sliceSlider.value()
		ls_index = self.loadstepSlider.value()
		sliceIndex = self.sliceCombo.currentIndex()
		if (sliceIndex == 0):
			data = field.data[ls_index][field.component,s_index,:,:]
		elif (sliceIndex == 1):
			data = field.data[ls_index][field.component,:,s_index,:]
		else:
			data = field.data[ls_index][field.component,:,:,s_index]
		return data

	def resetBounds(self):
		if (self.customBoundsCheck.checkState() == 0 and self.currentFieldIndex != None):

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
		
	def changeField(self, index, checked):

		if (self.changeFieldSuspended):
			return

		if checked:
			self.currentFieldIndex = index
			self.changeFieldSuspended = True
			for i, field in enumerate(self.fields):
				field.button.setChecked(i == index)
			self.changeFieldSuspended = False
			self.resetBounds()
			self.viewXMLButton.setChecked(False)
			self.viewResultDataButton.setChecked(False)
		else:
			self.currentFieldIndex = None

		self.replot()

	def sliceSliderChanged(self):
		self.resetBounds()
		self.replot()
		self.sliceLabel.setText("%s=%04d" % (self.sliceCombo.currentText(), self.sliceSlider.value()))

	def alphaSliderChanged(self):
		self.alphaLabel.setText("alpha=%4f" % self.getAlpha())
		self.customBoundsCheckChanged(0)

	def replot(self):

		self.axes.clear()

		xlim = None
		ylim = None

		if self.cb:
			if not self.replot_reset_limits:
				xlim = self.axes.get_xlim()
				ylim = self.axes.get_ylim()
			self.replot_reset_limits = False
			self.cb.ax.clear()
			cbax = self.cb.ax
		else:
			cbax = None
		
		if (self.currentFieldIndex != None):
			
			data = self.getCurrentSlice()
			s_index = self.sliceCombo.currentIndex()
			coords = ["x", "y", "z"]
			z_cord = coords[s_index]
			xy_cord = coords
			del xy_cord[s_index]

			vmin = float(self.vminText.text())
			vmax = float(self.vmaxText.text())
			
			#color_norm = matplotlib.colors.SymLogNorm(linthresh=1e-2, linscale=1)
			cm = mcmap.get_cmap(self.colormapCombo.currentText(), 2**11)

			p = self.axes.imshow(data.T, interpolation="nearest", origin="lower", norm=None, cmap=cm, vmin=vmin, vmax=vmax)

			if (cbax != None):
				self.cb = self.fig.colorbar(p, cax=cbax)
			else:
				self.cb = self.fig.colorbar(p, shrink=0.7)
		
			z_label = self.fields[self.currentFieldIndex].label
			self.cb.ax.set_title(z_label, y=1.03)

			numcols, numrows = data.shape
			def format_coord(x, y):
				col = int(x+0.5)
				row = int(y+0.5)
				s = '%s = %d, %s = %d' % (xy_cord[0], col, xy_cord[1], row)
				if col>=0 and col<numcols and row>=0 and row<numrows:
					z = data[col,row]
					return '%s, %s = %1.4f' % (s, z_label, z)
				else:
					return s
			def get_cursor_data(event):
				return None
			self.axes.format_coord = format_coord
			p.get_cursor_data = get_cursor_data

			font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.GeneralFont)
			self.axes.set_xlabel(xy_cord[0], labelpad=font.pointSize())
			self.axes.set_ylabel(xy_cord[1], labelpad=font.pointSize())

			self.figcanvas.setVisible(True)
			self.fignavbar.setVisible(True)
		else:
			self.figcanvas.setVisible(False)
			self.fignavbar.setVisible(False)

		if (xlim != None):
			self.axes.set_xlim(xlim)
		if (ylim != None):
			self.axes.set_ylim(ylim)

		self.axes.xaxis.set_major_formatter(mtick.FormatStrFormatter('%04d'))
		self.axes.yaxis.set_major_formatter(mtick.FormatStrFormatter('%04d'))

		self.figcanvas.draw()


class XMLHighlighter(QtGui.QSyntaxHighlighter):
 
	def __init__(self, parent=None):
		super(XMLHighlighter, self).__init__(parent)
 
		self.highlightingRules = []

		app = QtWidgets.QApplication.instance()
		pal = app.palette()

		xmlElementFormat = QtGui.QTextCharFormat()
		xmlElementFormat.setFontWeight(QtGui.QFont.Bold)
		xmlElementFormat.setForeground(QtCore.Qt.darkGreen)
		self.highlightingRules.append((QtCore.QRegExp("<[/\s]*[A-Za-z0-9_]+[\s/>]+"), xmlElementFormat))
 
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
		self.highlightingRules.append((QtCore.QRegExp("\\b[A-Za-z0-9_]+(?=\\=)"), xmlAttributeFormat))
 
		valueFormat = QtGui.QTextCharFormat()
		valueFormat.setForeground(pal.windowText().color())
		self.highlightingRules.append((QtCore.QRegExp("['\"][^'\"]*['\"]"), valueFormat))

		self.commentFormat = QtGui.QTextCharFormat()
		self.commentFormat.setForeground(QtCore.Qt.gray)
		self.commentStartExpression = QtCore.QRegExp("<!--")
		self.commentEndExpression = QtCore.QRegExp("-->")

	def highlightBlock(self, text):
		
		#for every pattern
		for pattern, format in self.highlightingRules:
 
			#Create a regular expression from the retrieved pattern
			expression = QtCore.QRegExp(pattern)
 
			#Check what index that expression occurs at with the ENTIRE text
			index = expression.indexIn(text)
 
			#While the index is greater than 0
			while index >= 0:
 
				#Get the length of how long the expression is true, set the format from the start to the length with the text format
				length = expression.matchedLength()
				self.setFormat(index, length, format)
 
				#Set index to where the expression ends in the text
				index = expression.indexIn(text, index + length)
 
		# handle comments
		self.setCurrentBlockState(0)
 
		startIndex = 0
		if self.previousBlockState() != 1:
			startIndex = self.commentStartExpression.indexIn(text)
		
		while startIndex >= 0:
			endIndex = self.commentEndExpression.indexIn(text, startIndex)
			commentLength = 0
			if endIndex == -1:
				self.setCurrentBlockState(1)
				commentLength = len(text) - startIndex
			else:
				commentLength = endIndex - startIndex + self.commentEndExpression.matchedLength()
			self.setFormat(startIndex, commentLength, self.commentFormat)
			startIndex = self.commentStartExpression.indexIn(text, startIndex + commentLength)



class XMLTextEdit(QtWidgets.QTextEdit):

	def __init__(self, parent = None):
		QtWidgets.QTextEdit.__init__(self, parent)
		font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
		#font = QtGui.QFont()
		#font.setFamily("Monospace")
		#font.setStyleHint(QtGui.QFont.Monospace)
		font.setFixedPitch(True)
		font.setPointSize(font.pointSize())
		fontmetrics = QtGui.QFontMetrics(font)
		self.setFont(font)
		self.setTabStopWidth(2 * fontmetrics.width(' '))
		self.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)

		# add syntax highlighting
		self.highlighter = XMLHighlighter(self.document())


class HelpWidget(QtWebKitWidgets.QWebView):

	def __init__(self, editor, parent = None):

		QtWebKitWidgets.QWebView.__init__(self, parent)

		self.editor = editor
		self.editor.selectionChanged.connect(self.editorSelectionChanged)
		self.editor.textChanged.connect(self.editorSelectionChanged)
		self.editor.cursorPositionChanged.connect(self.editorSelectionChanged)

		self.timer = QtCore.QTimer(self)
		self.timer.setInterval(100)
		self.timer.timeout.connect(self.updateHelp)

		cdir = os.path.dirname(os.path.abspath(__file__))

		self.ff = ET.parse(cdir + "/../doc/fileformat.xml")
		
		self.mypage = MyWebPage()
		self.mypage.linkClicked.connect(self.linkClicked)
		self.setPage(self.mypage)

		#self.setStyleSheet("background:transparent");
		#self.setAttribute(QtCore.Qt.WA_TranslucentBackground);

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

		if url[1] == "add":
			if url[3] == "empty":
				c.insertText(indent + "<" + url[2] + " />")
				c.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.MoveAnchor, 3)
			else:
				c.insertText(indent + "<" + url[2] + ">" + url[4] + "</" + url[2] + ">")
				c.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.MoveAnchor, len(url[2]) + 3)
				c.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.KeepAnchor, len(url[4]))
				
		elif url[1] == "set":
			pos1 = int(url[5])
			if txt[pos1-2] == "/":
				c.setPosition(pos1-2)
			else:
				c.setPosition(pos1-1)
			ins = url[2] + '="' + url[3] + '"'
			if (txt[c.position()-1].strip() != ""):
				ins = " " + ins
			if (txt[c.position()].strip() != ""):
				ins += " "
			c.insertText(ins)
			c.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.MoveAnchor, 1)
			c.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.KeepAnchor, len(url[3]))
		elif url[1] == "ins":
			txt = url[2]
			pos1 = int(url[4])
			pos2 = txt.find("<", pos1)
			if (pos2 >= 0):
				c.setPosition(pos1)
				c.movePosition(QtGui.QTextCursor.Right, QtGui.QTextCursor.KeepAnchor, pos2-pos1)
			c.insertText(txt)
			c.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.KeepAnchor, len(txt))

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
					break
				e = en

		typ = e.get("type")
		values = e.get("values")

		html = self.mypage.defaultCSS() + """
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

		path = ".".join(i[0] for i in items)
		html += "<h2>" + path + "</h2>"

		
		if en is None:
			help = "Unknown element"
		else:
			help = e.get("help")

		html += '<div class="help">' + help + "</div>"

		if en is None:
			pass
		elif inside or typ != "list":

			if typ != "none":
				html += "<p><b>Type:</b> " + typ + "</p>"

			if typ == "bool":
				values = "0,1"

			if not values is None:
				values = values.split(",")
				values = sorted(values)
				html += "<p><b>Valid values:</b> "
				for i, v in enumerate(values):
					if i > 0:
						html += " | "
					html += '<a href="http://x#ins#' + v + '#' + str(item[1].start()) + '#' + str(item[1].end()) + '">' + v + '</a>'
				html += "</p>"

			if not e.text is None and len(e.text.strip()) > 0:
				html += '<p><b>Default:</b> ' + e.text.strip() + "</p>"
			
			if (not en is None):
				attr = ""
				attribs = list(e.findall("attrib"))
				attribs = sorted(attribs, key=lambda a: a.get("name"))
				for a in attribs:
					default = ("" if a.text is None else a.text.strip())
					attr += "<tr>"
					attr += '<td><b><a href="http://x#set#' + a.get("name") + '#' + default + '#' + str(item[1].start()) + '#' + str(item[1].end()) + '">' + a.get("name") + "</a></b></td>"
					attr += "<td>" + a.get("type") + "</td>"
					attr += "<td>" + default + "</td>"
					attr += "<td>" + a.get("help") + "</td>"
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
			items = sorted(items, key=lambda e: e.tag)
			for a in items:
				if a.tag == "attrib":
					continue
				typ = a.get("type")
				default = ("" if a.text is None else a.text.strip())
				tags += "<tr>"
				tags += '<td><b><a href="http://x#add#' + a.tag + '#' + typ + '#' + default + '">' + a.tag + "</a></b></td>"
				tags += "<td>" + typ + "</td>"
				tags += "<td>" + default + "</td>"
				tags += "<td>" + a.get("help") + "</td>"
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

		# ui.textEdit->verticalScrollBar()->setValue(0);
		self.mypage.setHtml(html)


class DocWidget(QtWebKitWidgets.QWebView):

	def __init__(self, parent = None):

		QtWebKitWidgets.QWebView.__init__(self, parent)

		cdir = os.path.dirname(os.path.abspath(__file__))
		self.docfile = os.path.abspath(os.path.join(cdir, "../doc/manual.html"))

		"""
		with open(self.docfile, "rt") as f:
			html = f.read()
		html = html.replace("<body>", "<body>" + self.mypage.defaultCSS())
		self.mypage.setHtml(html)
		"""

		self.mypage = MyWebPage()
		self.mypage.setUrl(QtCore.QUrl("file://" + self.docfile))
		self.mypage.linkClicked.connect(self.linkClicked)
		self.setPage(self.mypage)

		css = self.mypage.defaultCSS(False)
		css = """
body {
	background-color: white;
}
"""
		data = base64.b64encode(css.encode('utf8')).decode('ascii')
		self.settings().setUserStyleSheetUrl(QtCore.QUrl("data:text/css;charset=utf-8;base64," + data))

	def linkClicked(self, url):
		self.mypage.setUrl(url)


class DemoWidget(QtWebKitWidgets.QWebView):

	openProjectRequest = QtCore.pyqtSignal('QString')
	newProjectRequest = QtCore.pyqtSignal('QString')

	def __init__(self, parent = None):

		QtWebKitWidgets.QWebView.__init__(self, parent)

		cdir = os.path.dirname(os.path.abspath(__file__))
		self.demodir = os.path.abspath(os.path.join(cdir, "../demo"))

		self.mypage = MyWebPage()
		self.mypage.linkClicked.connect(self.linkClicked)
		self.setPage(self.mypage)

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

		html = self.mypage.defaultCSS() + """
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
	background-color: initial;
	padding: 1em;
	border-bottom: 1px solid """ + pal.shadow().color().name() + """;
	margin-bottom: 1em;
}
</style>
"""

		category_file = os.path.join(path, "category.xml")

		if os.path.isfile(category_file):
			try:
				xml = ET.parse(category_file).getroot()
				html += '<table class="header">'
				html += '<tr>'
				if path == self.demodir:
					html += '<td>'
					html += '<h1>fibergen</h1>'
					html += '<p>A FFT-based homogenization tool.</p>'
					html += '</td>'
					img = xml.find("image")
					if not img is None and not img.text is None and len(img.text):
						img = os.path.join(path, img.text)
						html += '<td><img src="file://' + img + '" /></td>'
				else:
					html += '<td>'
					title = xml.find("title")
					if not title is None and len(title.text):
						html += '<h1>' + title.text + '</h1>'
					else:
						html += '<h1>' + d + '</h1>'
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
		for d in os.listdir(path):
			subdir = os.path.join(path, d)
			if not os.path.isdir(subdir):
				continue

			project_file = os.path.join(subdir, "project.xml")
			category_file = os.path.join(subdir, "category.xml")

			item = ""
			index = None
			if os.path.isfile(project_file):
				try:
					xml = ET.parse(project_file).getroot()
				except:
					print("error in file", project_file)
					print(traceback.format_exc())
					continue
				try:
					action = xml.find("action").text
				except:
					action = "new" if d == "empty" else "open"
				item += '<a class="demo" href="http://x#' + action + '#' + project_file + '">'
				title = xml.find("title")
				if not title is None and not title.text is None and len(title.text):
					item += '<h2>' + title.text + '</h2>'
				else:
					item += '<h2>' + d + '</h2>'
				img = xml.find("image")
				if not img is None and not img.text is None and len(img.text):
					img = os.path.join(subdir, img.text)
					item += '<img src="file://' + img + '" />'
				else:
					for ext in ["svg", "png"]:
						img = os.path.join(subdir, "thumbnail." + ext)
						if os.path.isfile(img):
							item += '<img src="file://' + img + '" />'
							break
				desc = xml.find("description")
				if not desc is None and not desc.text is None and len(desc.text):
					item += '<p>' + desc.text + '</p>'
				item += '</a>'
				index = xml.find("index")
			elif os.path.isfile(category_file):
				try:
					xml = ET.parse(category_file).getroot()
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
					item += '<img src="file://' + img + '" />'
				item += '</a>'
				index = xml.find("index")
			else:
				continue

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

		self.mypage.setHtml(html)


class MainWindow(QtWidgets.QMainWindow):
 
	def __init__(self, parent = None):
		
		app = QtWidgets.QApplication.instance()
		pal = app.palette()

		QtWidgets.QMainWindow.__init__(self, parent)
 
		#self.setMinimumSize(1000, 800)
		dir_path = os.path.dirname(os.path.realpath(__file__))
		self.setWindowTitle("FFT Homogenization Tool")
		self.setWindowIcon(QtGui.QIcon(dir_path + "/../gui/icons/logo1/icon32.png"))


		self.textEdit = XMLTextEdit()
	
		self.runConut = 0
		self.lastSaveText = self.getSaveText()

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
		self.tabWidget.tabCloseRequested.connect(self.tabCloseRequested)
		
		self.filename = None
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
				break

		# add toolbar actions

		def aa(icon, text, func, key):
			action = self.toolbar.addAction(QtGui.QIcon.fromTheme(icon), text)
			action.triggered.connect(func)
			action.setShortcut(key)
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
			self.hSplit.restoreState(app.settings.value("hSplitterSize"))
			self.vSplit.restoreState(app.settings.value("vSplitterSize"))
		except:
			print(traceback.format_exc())
			screen = app.desktop().screenGeometry()
			self.resize(screen.width()*2/3, screen.height()*2/3)
			#self.setWindowState(QtCore.Qt.WindowMaximized)
			self.hSplit.setSizes([self.width()/3, 2*self.width()/3])
			self.vSplit.setSizes([2*self.height()/3, self.height()/3, 1])

		self.setDocumentVisible(False)
		self.tabWidget.setVisible(False)

		self.show()

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
		webbrowser.open('https://fospald.github.io/fibergen/')

	def openHelp(self):
		if self.docTabIndex is None:
			if self.docTab is None:
				self.docTab = DocWidget()
			self.docTabIndex = self.addTab(self.docTab, "Help")
		self.tabWidget.setCurrentIndex(self.docTabIndex)

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
		if index == self.demoTabIndex:
			self.demoTabIndex = None
		elif index == self.docTabIndex:
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
			self.updateStatus()
			return True
		except:
			QtWidgets.QMessageBox.critical(self, "Error", sys.exc_info()[0])
			return False
	
	def openDemo(self, filename):
		if not self.openProjectSave(filename):
			return False
		if not self.demoTabIndex is None:
			self.tabCloseRequested(self.demoTabIndex)
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
			self.file_id += 1
			self.lastSaveText = self.getSaveText()
			self.textEdit.document().clearUndoRedoStacks()
			self.setDocumentVisible(True)
			self.updateStatus()
		except:
			QtWidgets.QMessageBox.critical(self, "Error", sys.exc_info()[0])
			return False
		return True

	def newProjectGui(self):
		if self.demoTabIndex is None:
			if self.demoTab is None:
				self.demoTab = DemoWidget()
				self.demoTab.openProjectRequest.connect(self.openDemo)
				self.demoTab.newProjectRequest.connect(self.newProject)
			self.demoTabIndex = self.addTab(self.demoTab, "Demos")
		self.tabWidget.setCurrentIndex(self.demoTabIndex)

	def newProject(self, filename=""):
		if not self.checkTextSaved():
			return False
		if not self.demoTabIndex is None:
			self.tabCloseRequested(self.demoTabIndex)
			self.demoTabIndex = None
		txt = ""
		try:
			with open(filename, "rt") as f:
				txt = f.read()
		except:
			pass
		self.textEdit.setPlainText(txt)
		self.filename = None
		self.file_id += 1
		self.lastSaveText = self.getSaveText()
		self.textEdit.document().clearUndoRedoStacks()
		self.setDocumentVisible(True)
		self.updateStatus()
		return True

	def runProject(self, fg=None):

		if not isinstance(fg, fibergen.FG):
			try:
				fg = fibergen.FG().init()
				xml = str(self.textEdit.toPlainText())
				fg.set_xml(xml)
			except:
				print(traceback.format_exc())
		else:
			xml = fg.get_xml()
			self.textEdit.setPlainText(xml)
		
		coord_names1 = ["x", "y", "z"]
		coord_names2 = ["xx", "yy", "zz", "yz", "xz", "xy", "zy", "zx", "yx"]

		field_labels = {"phi": lambda i: u"φ_%d" % i,
			"epsilon": lambda i: u"ε_%s" % coord_names2[i],
			"sigma": lambda i: u"σ_%s" % coord_names2[i],
			"u": lambda i: "u_%s" % coord_names1[i],
			"normals": lambda i: "n_%s" % coord_names1[i],
			"orientation": lambda i: "d_%s" % coord_names1[i],
			"fiber_translation": lambda i: "t_%s" % coord_names1[i],
			"fiber_id": lambda i: "id",
			"p": lambda i: "p",
		}

		phase_fields = ["phi"]
		run_fields = ["epsilon", "sigma", "u"]
		const_fields = ["phi", "fiber_id", "fiber_translation", "normals", "orientation"]

		field_names = phase_fields + run_fields

		try:
			mode = fg.get("solver.mode".encode('utf8'))
		except:
			mode = "elasticity"

		if (mode == "viscosity"):
			field_labels["epsilon"] = lambda i: "σ_%s" % coord_names2[i]
			field_labels["sigma"] = lambda i: "ɣ_%s" % coord_names2[i]
			run_fields.append("p")

		if (mode == "heat"):
			field_labels["u"] = lambda i: "T"
			field_labels["epsilon"] = lambda i: "∇T_%s" % coord_names1[i]
			field_labels["sigma"] = lambda i: "q_%s" % coord_names1[i]

		field_groups = []
		mean_strains = []
		mean_stresses = []
		loadstep_called = []

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

			if (len(field_groups) == 0):
				for name in field_names:
					data = fg.get_field(name.encode('utf8'))
					shape = data.shape
					fields = []
					for i in range(shape[0]):
						field = PlotField()
						field.data = [data]
						field.label = field_labels[name](i)
						field.name = name
						field.component = i
						#field.amin = np.amin(data[i])
						#field.amax = np.amax(data[i])
						fields.append(field)
					field_groups.append(fields)
			else:
				for field_group in field_groups:
					for field in field_group:
						if field.name in const_fields:
							data = field.data[-1]
						else:
							data = fg.get_field(field.name.encode('utf8'))
						field.data.append(data)

			process_events()

		def convergence_callback():
			#residual = fg.get_residuals()[-1]
			#print "res=", residual
			# progress.setValue(100)
			process_events()
			mean_strains.append(fg.get_mean_strain())
			mean_stresses.append(fg.get_mean_stress())
			process_events()
			return progress.wasCanceled()


		try:
			fg.set_loadstep_callback(loadstep_callback)
			fg.set_convergence_callback(convergence_callback)
			#print fg.get_xml()

			progress.show()
			process_events()
			
			fg.run()
			fg.init_phase()

			if len(loadstep_called) == 0:
				field_names = phase_fields
				loadstep_callback()

			if progress.wasCanceled():
				progress.close()
				return

		except:
			print(traceback.format_exc())

		progress.close()
		process_events()
		
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
				if isinstance(r, collections.Iterable):
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
			('effective_property', matrix(mat(fg.get_effective_property()))),
		]))

		if len(residuals) > 0:
			resultText += section('Residuals')
			resultText += table(collections.OrderedDict([
				('residuals', matrix([[i,r] for i,r in enumerate(residuals)])),
			]))
		
		"""
		for i, ij in enumerate([11, 22, 33, 23, 13, 12]):
			resultText += plot(range(len(mean_stresses)), [s[i] for s in mean_stresses], "Sigma_%s" % ij, "Iteration", "Sigma_%d" % ij, "linear")
			resultText += plot(range(len(mean_strains)), [s[i] for s in mean_strains], "Epsilon_%s" % ij, "Iteration", "Epsilon_%d" % ij, "linear")
		"""

		other = self.tabWidget.currentWidget()

		if not isinstance(other, PlotWidget):
			other = None
		elif other.file_id != self.file_id:
			other = None

		tab = PlotWidget(field_groups, xml, resultText, other)
		tab.file_id = self.file_id
		if len(tab.fields) > 0:
			i = self.addTab(tab, "Run_%d" % self.runConut)
		self.runConut += 1


class App(QtWidgets.QApplication):

	def __init__(self, args):

		QtWidgets.QApplication.__init__(self, list(args) + ["--disable-web-security"])

		# set matplotlib defaults
		font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.GeneralFont)
		pal = self.palette()
		text_color = pal.text().color().name()
		bg_color = pal.base().color().name()
		rcParams.update({
			'figure.autolayout': True,
			'font.size': font.pointSize(),
			'font.family': font.family(),
			'text.color': text_color,
			'axes.labelcolor': text_color,
			'xtick.color': text_color,
			'ytick.color': text_color,
			#'figure.facecolor': bg_color,
			#'savefig.facecolor': bg_color,
			#'axes.facecolor': bg_color,
			'backend': 'Qt5Agg',
		})

		self.settings = QtCore.QSettings("NumaPDE", "FIBERGEN")
		self.window = MainWindow()

		try:
			if (len(args) > 1):
				self.window.openProject(args[-1])
				#self.window.runProject()
			else:
				self.window.newProjectGui()
		except:
			print(traceback.format_exc())

	def notify(self, receiver, event):
		try:
			QtWidgets.QApplication.notify(self, receiver, event)
		except e:
			QtWidgets.QMessageBox.critical(self, "Error", sys.exc_info()[0])
		return False
	
	def restoreWindowState(self, win, prefix):
		win.restoreGeometry(self.settings.value(prefix + "_geometry"))
		if (isinstance(win, QtWidgets.QMainWindow)):
			win.restoreState(self.settings.value(prefix + "_windowState"))

	def saveWindowState(self, win, prefix):
		self.settings.setValue(prefix + "_geometry", win.saveGeometry())
		if (isinstance(win, QtWidgets.QMainWindow)):
			self.settings.setValue(prefix + "_windowState", win.saveState())

 
if __name__ == "__main__":
	app = App(sys.argv)
	ret = app.exec_()
	sys.exit(ret)

