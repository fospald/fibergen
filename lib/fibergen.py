
import socket, os, subprocess

#os.system("fibergen-build")

hostname = subprocess.check_output(['uname', '-n']).decode("utf-8").strip()
#hostname = socket.gethostname()
for c in ["-", ".", " "]:
	hostname = hostname.replace(c, "_")

libname = "fibergen_" + hostname

fibergen = __import__(libname)
locals().update(fibergen.__dict__)

#fibergen_gui = __import__("fibergen_gui")
#locals().update({'gui': fibergen_gui.__dict__})

