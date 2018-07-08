
import socket, os

#os.system("fibergen-build")

hostname = socket.gethostname()
hostname = hostname.replace("-", "_")
hostname = hostname.replace(" ", "_")

libname = "fibergen_" + hostname

fibergen = __import__(libname)
locals().update(fibergen.__dict__)

#fibergen_gui = __import__("fibergen_gui")
#locals().update({'gui': fibergen_gui.__dict__})


