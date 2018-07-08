#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fibergen_gui as fg_gui
import sys

if __name__ == "__main__":
	app = fg_gui.App(sys.argv)
	sys.exit(app.exec_())

