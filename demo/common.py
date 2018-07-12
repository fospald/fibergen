from __future__ import division
import os
import numpy as np
import matplotlib as mp

#if not "DISPLAY" in os.environ or os.environ["DISPLAY"] == "":
#	mp.use('Agg')
#else:
#	mp.use('Agg')
	#mp.use('Qt4Agg') # [u'pgf', u'cairo', u'MacOSX', u'CocoaAgg', u'gdk', u'ps', u'GTKAgg', u'nbAgg', u'GTK', u'Qt5Agg', u'template', u'emf', u'GTK3Cairo', u'GTK3Agg', u'WX', u'Qt4Agg', u'TkAgg', u'agg', u'svg', u'GTKCairo', u'WXAgg', u'WebAgg', u'pdf']

import matplotlib.pyplot as plt
import matplotlib.markers
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
import fibergen
import hashlib
import sys, math
import pickle
import csv
from collections import *
import mpi4py.MPI
import multiprocessing

np.set_printoptions(linewidth=999999, precision=3)

# enable latex in plots
#import matplotlib.rc
#mp.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#mp.rc('text', usetex=True)

# init random seed to fixed value
np.random.seed(0)

#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#plt.rc('text', usetex=True)


def run_tasks_parallel(tasks):
	for i, task in enumerate(tasks):
		rank = mpi4py.MPI.COMM_WORLD.Get_rank()
		size = mpi4py.MPI.COMM_WORLD.Get_size()
		if i%size!=rank: continue
		print("Task number %d being done by processor %d of %d" % (i, rank, size))
		task()

def run_tasks(tasks):
	for i, task in enumerate(tasks):
		task()


class ResultDict(OrderedDict):
	def __getitem__(self, key):
		if not key in self:
			self[key] = []
		return OrderedDict.__getitem__(self, key)


class Experiment(object):

	def __init__(self, project_xml, results_dat=None):
		self.project_xml = project_xml
		self.results_dat = results_dat
		self.parameters  = []
		self.info  = []
		self.results     = []
		self.fiber_file  = None
		self.fiber_group = None

	def add_info(self, path, value):
		print("+info", path, value)
		self.parameters.append(([path], [value], ['info']))

	def add_param(self, path, values, record=None):
		return self.add_params(path, values, record)

	def add_params(self, path, values, record=None):
		if not isinstance(values, MutableSequence):
			values = [values]
			if record is None:
				record = [False]
		if not isinstance(path, MutableSequence):
			path = [path]
		if len(path) == 1 and len(values) > 1:
			path = path*len(values)
		if not isinstance(record, MutableSequence):
			if record is None:
				record = True
			record = [record]
			if len(record) == 1:
				record = record*len(values)
		self.parameters.append((path, values, record))

	def add_result(self, key, record=None):

		key_map = {
			"solve_time":  (fg.get_solve_time, True),
			"mean_stress": (fg.get_mean_stress, True),
			"mean_strain": (fg.get_mean_strain, True),
			"A2": (fg.get_A2, True),
			"A4": (fg.get_A4, True),
			"iterations":  (lambda: len(fg.get_residuals()), True),
			"residuals":   (fg.get_residuals, False),
		}

		if (key in key_map):
			func, rec = key_map[key]
			if record is None:
				record = rec
		else:
			raise RuntimeError("key not found")

		self.results.append((key, func, record))

	def add_results(self, keys, record=None):

		if not isinstance(keys, MutableSequence):
			keys = [keys]

		for key in keys:
			self.add_result(key, record)

	def set_fibers(self, fiber_file, fiber_group="group-fibers"):
		self.fiber_file  = fiber_file
		self.fiber_group = fiber_group

	def create_assignment(self, key, value):

		if isinstance(key, MutableSequence):
			assignments = []
			for i, k in enumerate(key):
				assignments += self.create_assignment(k, value[i])
			return assignments

		key_map = {
			"resolution": "solver..n",
			"resolution_x": "solver..nx",
			"resolution_y": "solver..ny",
			"resolution_z": "solver..nz",
			"dim_x": "dx",
			"dim_y": "dy",
			"dim_z": "dz",
			"smooth_levels": "solver.smooth_levels",
			"tol": "solver.tol",
			"num_threads": "num_threads",
			"num_fibers": "n",
			"fiber_length": "length",
			"fiber_radius": "radius",
			"seed": "seed",
		}

		if (key in key_map):
			path = key_map[key]
		else:
			path = key
			#raise RuntimeError("key not found")

		return [(key, path, value)]
	
	def voigt_index_keygen(self, key, i):
		return "%s_%d" % (key, [11, 22, 33, 23, 13, 12, 32, 31, 21][i])

	def expand_data(self, key, data):

		mode = fg.get("solver.mode")
		if mode == "elasticity" or mode == "hyperelasticity":
			key_map = {
				'mean_stress': "sigma",
				'mean_strain': "epsilon",
			}
		elif mode == "viscosity":
			key_map = {
				'mean_stress': "gamma",
				'mean_strain': "sigma",
			}

		if key in key_map:
			key = key_map[key]

		if isinstance(data, MutableSequence):
			items = []
			keygen = lambda i: self.voigt_index_keygen(key, i)
			for i, d in enumerate(data):
				items.append((keygen(i), d))
			return items
		else:
			return [(key, data)]

	def run(self, version=0, dry=False, res=ResultDict(), id_prefix=[], cache_only=False):

		if not isinstance(id_prefix, MutableSequence):
			id_prefix = [id_prefix]

		indices = [0]*len(self.parameters)

		while True:

			row = ResultDict()

			# create list of assignments
			assignments = []
			current_id = list(id_prefix)
			for j, param in enumerate(self.parameters):

				path, values, record = param

				values = values[indices[j]]
				path = path[indices[j]]
				record = record[indices[j]]

				if (record == 'info'):
					row[path].append(values)
					continue
				
				# value_i = values[i] if isinstance(values, MutableSequence) else values
				# print "##", path, values
				a = self.create_assignment(path, values)
				assignments += a

				# append to results
				if record:
					for key, path, value in a:
						row[key].append(value)

				if len(param[1]) > 1:
					current_id += [indices[j]]
			
			# run experiment
			data = self.run_experiment(assignments, self.results, version, dry, cache_only)
			
			if not data is None:

				current_id = "_".join(map(str, current_id))
				row["id"].append(current_id)

				# add results
				for key, func, record in self.results:
					if not record:
						continue
					key_values = self.expand_data(key, data[key])
					for key, value in key_values:
						row[key].append(value)
				
				# append results
				for key in row.keys():
					if not key in res:
						res[key] = []
					res[key].append(row[key])

			# increase index
			carry = 1
			for i in reversed(range(len(self.parameters))):
				if carry == 0:
					break
				indices[i] += carry
				if (indices[i] >= len(self.parameters[i][1])):
					indices[i] = 0
					carry = 1
				else:
					carry = 0
		
			# check if we looped over all indices
			if (carry == 1):
				break
		
		# write results to file
		if not self.results_dat is None:
			self.write_dict(self.results_dat, res)
	
		return res

	@staticmethod
	def write_dict(filename, res):
		def extact_value(v):
			return np.array(v).item();
		print("writing", filename)
		with open(filename, "wb+") as f:
			if (len(res.keys()) > 0):
				f.write("%s\n" % "\t".join(res.keys()))
				for i in range(len(res[res.keys()[0]])):
					f.write("%s\n" % "\t".join([str(extact_value(res[k][i])) for k in res.keys()]))

	def assign_param(self, key, value):
		if hasattr(value, '__call__'):
			value = value()
		print("+param", key, value)
		if isinstance(value, MutableSequence):
			if isinstance(value, dict):
				fg.set(key, **value)
			else:
				fg.set(key, *value)
		else:
			fg.set(key, value)
	
	def run_experiment(self, assignments, results, version, dry, cache_only):
		
		print("##### experiment")

		def experiment_step():

			for key, path, value in assignments:
				self.assign_param(path, value)

			run_fg()

			res = dict()
			for key, func, report in results:
				res[key] = func()
		
			return res


		fg.reset()
		fg.set_xml_precision(5)
		fg.load_xml(self.project_xml)
		fg.set("results", ",".join([r[0] for r in results]))

		if not self.fiber_file is None:
			self.add_fibers()

		for key, path, value in assignments:
			self.assign_param(path, value)

		#with open("last_project.xml", "wb+") as f:
		#	f.write(fg.get_xml())

		fg.erase("num_threads")
		fg.erase("solver.fft_planner_flag")

		print(fg.get_xml())
		#return

		if dry:
			data = dict()
			for key, func in results:
				data[key] = "dry"
			return data

		data = run_experiment(experiment_step, version, cache_only=cache_only)

		return data


	def add_fibers(self):
		with open(self.fiber_file, 'rb') as csv_file:
			csv_reader = csv.reader(csv_file, delimiter='\t')
			nfibers = 0
			for row in csv_reader:
				# row = Cx Cy Cz Ax Ay Az R L
				row = map(float, row)
				L = row[7]
				R = row[6]
				L0 = L + (4.0/3.0)*R
				fg.set("actions.%s.place_fiber[%d]." % (self.fiber_group, nfibers),
					cx=row[0], cy=row[1], cz=row[2], ax=row[3], ay=row[4], az=row[5], R=R, L=L0)
				nfibers += 1

	# returns a lambda function for calculating the number of smooth levels based on the resolution
	def smooth_level_calc(self, max_refinements, resolution_key="solver..n"):
		def func(max_refinements, resolution_key):
			n = int(fg.get(resolution_key))
			refinement = np.round(np.log(n)/np.log(2))
			return int(max_refinements-refinement)
		return lambda: func(max_refinements, resolution_key)


class IsoSpherePointGenerator(object):
	# create equal distanced integration points on the sphere
	@staticmethod
	def generate(num_points, cache_only=False):
		# projection of points to sphere
		def project(points):
			points_norm = np.linalg.norm(points, ord=2, axis=0)
			points /= points_norm[np.newaxis, :]
			return (points, points_norm)
		# compute potential energy of points
		def potential_energy(points, s):
			# compute (chord) distance between all points
			dist = points[:, np.newaxis, :] - s*points[:, :, np.newaxis]
			dist_norm = np.linalg.norm(dist, ord=2, axis=0)
			# compute great circle distance
			# dist_norm = 2*np.arcsin(np.minimum(0.5*dist_norm, 1.0))
			# set the diagonal to infinity to ignored zero distances
			if (s == 1):
				np.fill_diagonal(dist_norm, np.Infinity)
			# compute potential energy
			energy = np.sum(1/dist_norm)
			if (np.isnan(energy)):
				# this may happen if points contain a point p and -p
				# since they are randomly generated, this is very unlikely
				raise "problem here"
			# compute the jacobian
			num_points = points.shape[1]
			jac = np.zeros_like(points)
			for i in range(num_points):
				w = points[:,i]
				p = np.sum((points - s*w[:,np.newaxis])/(dist_norm[i,:]**2), axis=1)
				jac[:,i] = p - np.dot(p,w)*w
			jac *= 4*s
			# return the energy and jacobian
			return (energy, jac)
		# objective and jacobian for minimization of the potential energy	
		def objective(x):
			# unravel points
			points, points_norm = project(x.reshape((3,-1)))
			num_points = points.shape[1]
			# compute relaxation factor for jacobian
			relax = 1/num_points
			# compute energy and jacobian of the point set (points u -points)
			energy1, jac1 = potential_energy(points, +1)
			energy2, jac2 = potential_energy(points, -1)
			energy = energy1 + energy2
			jac = relax*(jac1 + jac2)
			print("energy:", energy)
			return (energy, jac.ravel())
		# check cache file
		filename = ".points_%d.txt.gz" % num_points
		if (os.path.isfile(filename)):
			# load the points
			points, points_norm = project(np.loadtxt(filename).reshape(3,-1))
			return points
		elif cache_only:
			return None
		# create initial random points on sphere
		points = np.random.randn(3, num_points)
		points, points_norm = project(points)
		# minimize potential energy of points
		x0 = points.ravel()
		x = opt.minimize(objective, x0, method='CG', jac=True).x
		# unravel points
		points, points_norm = project(x.reshape((3,-1)))
		# save the points
		np.savetxt(filename, points)
		return points


def run_experiment(func, version_minor=0, verbose=True, cache_only=False):

	# check if caching is disabled
	if (version_minor < 0):
		return func()

	# get simulation parameters
	xml = fg.get_xml()

	# create hash code of parameters
	m = hashlib.md5()
	m.update(xml)
	hash_code = m.hexdigest()

	version_major = 0
	name = (func.__name__, version_minor, hash_code)

	simdir = get_simulation_dir()
	filebase = os.path.join(simdir, '_'.join([str(i) for i in name]))
	cache_txt = filebase + ".out"
	simfile = filebase + ".xml"

	if (verbose):
		print("XML-FILE:", simfile)

	try:
		with open(cache_txt, "rb") as f:
			data = pickle.load(f)
		if (verbose):
			print("Using cache for", name)
		if (isinstance(data, dict)):
			data['cache_filename'] = cache_txt
		return data
	except:
		pass

	if cache_only:
		return None

	# save simulation parameters to file
	with open(simfile, 'wb+') as f:
		f.write(xml)

	# run simulation
	data = func()

	#if (fg.get_error()):
	#	return None

	# save simulation results to file
	with open(cache_txt, "wb+") as f:
		pickle.dump(data, f)

	return data


def reset_fg(filename='project.xml'):
	fg.reset()
	fg.load_xml(filename)


def run_fg():
	fg.run()


def set_params_by_contrast(gamma, k_1_ge_1 = False):

	r_2 = 0.4		# radius of outher sphere (phase 2)
	r_1 = r_2*(0.5**(1/3.))	# radius of inner sphere (phase 1)
	r_1 = 0.2		# radius of outher sphere (phase 2)

	f_1 = (r_1**3)/(r_2**3)	# volume fraction occupied by phase 2 in the coated sphere
	f_2 = 1 - f_1		# volume fraction occupied by phase 2 in the coated sphere

	# compute bulk modulus of one phase from the other two phase materials
	# (required for continuity of the displacement field)
	# NOTE: bulkmodulus k = lambda + (2./3.)*mu
	# k_1 = k_2 + (k_3 - k_2)/(f_1 - f_2*(k_3 - k_2)/(k_2 + 4*mu_2/3.))
	# k_3 = k_2 + f_1*(k_1 - k_2)/(1 + f_2*(k_1 - k_2)/(k_2 + 4*mu_2/3.))
	# k_2 satisfies the quadratic equation:
	#  a*k_2**2 + b*k_2 + c = 0 with
	#  a = (f_1-1)
	#  b = (k_3 - 4*mu_2/3. - f_2 - f_1*k_1 + f_1*4*mu_2/3.)
	#  c = (4*mu_2/3.*k_3 + f_2*k_1 - f_1*k_1*4*mu_2/3.)
	# if we set mu_2 = k_2*3/5.0:
	#  a = 4/9.*f_2
	#  b = f_1*k_1 - k_3 + 5/9.*f_2*(k_1 + k_3)
	#  b = (1 - 4/9.*f_2)*k_1 - (1 - 5/9.*f_2)*k_3
	#  c = -5/9.*f_2*k_3*k_1

	#k_3 = k_2 + f_1*(k_1 - k_2)/(1 + f_2*(k_1 - k_2)/(k_2 + 4*mu_2/3.))
	k_3 = 1.0
	mu_3 = k_3*3/5.0	# matrix Lame parameters
	lambda_3 = mu_3

	if k_1_ge_1:
		k_1 = (-9*k_3 + 5*f_2*k_3 - 5*f_2*gamma*k_3)/(-4*f_2/gamma - 9 + 4*f_2)
	else:
		k_1 = (5*f_2*k_3/gamma + 9*k_3 - 5*f_2*k_3)/(9 - 4*f_2 + 4*f_2*gamma)
	#k_1 = k_2 + (k_3 - k_2)/(f_1 - f_2*(k_3 - k_2)/(k_2 + 4*mu_2/3.))
	#k_1 = 1.0
	mu_1 = k_1*3/5.0
	lambda_1 = mu_1

	q = (-9 + 4*f_2)*k_1 + (9-5*f_2)*k_3
	if (q < 0):
		k_2 = -10.0*f_2*k_1*k_3/((q - math.sqrt(80.0*f_2*f_2*k_1*k_3 + q*q)))	# numeric stable version for q < 0
	else:
		k_2 = 1.0/(8*f_2)*(q + math.sqrt(80.0*f_2*f_2*k_1*k_3 + q*q))
	mu_2 = k_2*3/5.0	# coating Lame parameters
	lambda_2 = mu_2

	#print gamma, (k_1, k_2, k_3)
	if k_1_ge_1:
		gamma_check = k_1/k_2
	else:
		gamma_check = k_2/k_1
	if abs(gamma - gamma_check)/gamma > 1e-5:
		#print gamma, gamma_check
		raise BaseException("check failed")

	fg.set("solver.materials.mat1..mu", mu_1);
	fg.set("solver.materials.mat1..lambda", lambda_1);
	fg.set("solver.materials.mat2..mu", mu_2);
	fg.set("solver.materials.mat2..lambda", lambda_2);
	fg.set("solver.materials.matrix..mu", mu_3);
	fg.set("solver.materials.matrix..lambda", lambda_3);

	if (False and k_1_ge_1):
		# set the reference material in this case
		mu_min = min(mu_1, mu_2, mu_3)
		lambda_min = min(lambda_1, lambda_2, lambda_3)
		ref = min(mu_min, lambda_min)
		fg.set("solver.materials.ref..mu", ref)
		fg.set("solver.materials.ref..lambda", ref)
		
	#print 'mu_1=', mu_1, ' lambda_1=', lambda_1, ' mu_2=', mu_2, ' lambda_2=', lambda_2, ' mu_3=', mu_3, ' lambda_3=', lambda_3
	#print "res = ", (k_2 - k_1 + (k_3 - k_2)/(f_1 - f_2*(k_3 - k_2)/(k_2 + 4*mu_2/3.)))
	#print "k_3 = ", k_3

	return {'mu_1': mu_1, 'lambda_1': lambda_1, 'k_1': k_1,
		'mu_2': mu_2, 'lambda_2': lambda_2, 'k_2': k_2,
		'mu_3': mu_3, 'lambda_3': lambda_3, 'k_3': k_3,
		'r_1': r_1, 'r_2': r_2}

def average_solve_time(f):
	sum_solve_time = 0.0
	t_measure = 2.0 if production else 0.0
	n = 0
	while True:
		data = f()
		sum_solve_time += data['solve_time']
		n += 1
		if sum_solve_time >= t_measure:
			break
	data['solve_time'] = sum_solve_time/n
	return data


def get_simulation_dir():
	simdir = os.path.join(get_result_dir(), "simulations")
	if not os.path.isdir(simdir):
		os.makedirs(simdir)
	return simdir


def get_result_dir():
	resdir = os.path.abspath(os.path.join("..", "results", "production" if production else "devel", os.path.basename(os.getcwd())))
	if not os.path.isdir(resdir):
		os.makedirs(resdir)
	return resdir


def vMisesStress(sigma):
	return np.sqrt(0.5*((sigma[0] - sigma[1])**2 + (sigma[1] - sigma[2])**2 + (sigma[2] - sigma[0])**2 + 6*(sigma[3]**2 + sigma[4]**2 + sigma[5]**2)))


def eoc(err, h, s = 1):
	return [s*math.log(err[i+1]/float(err[i]))/math.log(h[i+1]/float(h[i])) for i in range(len(h)-1)]

def memory():
	with open('/proc/meminfo', 'r') as mem:
		ret = {}
		tmp = 0
		for i in mem:
			sline = i.split()
			if str(sline[0]) == 'MemTotal:':
				ret['total'] = int(sline[1])*1024
			elif str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
				tmp += int(sline[1])*1024
		ret['free'] = tmp
		ret['used'] = int(ret['total']) - int(ret['free'])
	return ret

def print_max_problem_sizes():
	mem = memory()['free']
	print("free memory: %gGb" % (mem/(1024**3)))
	print("max problem size elasticity: %s (cg) %s (basic)" % (format_problem_size(max_problem_size(24,mem)), format_problem_size(max_problem_size(12,mem))))
	print("max problem size viscosity (staggered): %s (cg) %s (basic)" % (format_problem_size(max_problem_size(30,mem)), format_problem_size(max_problem_size(18,mem))))
	print("max problem size viscosity (collocated): %s (cg) %s (basic)" % (format_problem_size(max_problem_size(24,mem)), format_problem_size(max_problem_size(12,mem))))

def max_problem_size(number_of_variables, memfree):
	bytes_per_voxel = number_of_variables*8
	voxels = memfree / bytes_per_voxel
	return voxels

def format_problem_size(voxels):
	a = 2**math.floor(math.log(voxels**(1.0/3.0), 2))
	b = a
	c = math.floor(voxels/a**2)
	while (c > 2*a):
		c /= 2
		a *= 2
	if (c > a):
		d = a
		a = c
		c = b
		b = d
	return '%dx%dx%dx' % (a, b, c)

# save figure to filename
def savefig(filename, data = None):

	title = plt.gca().get_title()
	if not production and title == "":
		plt.title(filename)
	elif False:
		# remove title and [1] units
		plt.title("")
		ax = plt.gca()
		xlabel = ax.get_xlabel().replace(" [1]", "")
		ylabel = ax.get_ylabel().replace(" [1]", "")
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
	filename = filename if os.path.isabs(filename) else os.path.join(get_result_dir(), filename) 
	plt.savefig(filename)
	print("Saving", filename)
	plt.title(title)

	# save also data to a csv file
	if not (data is None):
		(root, ext) = os.path.splitext(filename)
		csv_filename = root
		savecsv(csv_filename, data)

# save dict of arrays to csv file
def savecsv(filename, data, keys = None, delimiter="\t"):
	
	print(filename)
	filename = filename if os.path.isabs(filename) else os.path.join(get_result_dir(), filename) 
	filename = filename + ".csv"
	
	with open(filename, 'wb+') as f:
		
		rows = []
		if isinstance(data, dict):
			if keys is None:
				keys = data.keys()
			num_rows = len(data[keys[0]])
			for i in range(num_rows):
				row = dict()
				for key in keys:
					row[key] = data[key][i]
				rows.append(row)
		else:
			for d in data:
				row = dict()
				for key in keys:
					if hasattr(d, "__getitem"):
						row[key] = d[key]
					else:
						row[key] = getattr(d, key)
				rows.append(row)
		
		keys = ["index"] + keys
		for i in range(len(rows)):
			rows[i]["index"] = i+1

		w = csv.DictWriter(f, keys, delimiter=delimiter)
		w.writeheader()
		for r in rows:
			w.writerow(r)

		print("Saving", filename)

def new_FG():
	fg = fibergen.FG()
	#fg.set_variable("fg", fg)
	return fg

print_max_problem_sizes()

# create fibergen instance
fg = new_FG()

# production mode (low res / high res switch)
production = True

# define some common stuff for plots
markers = matplotlib.markers.MarkerStyle.markers
markers = ['o', 's', 'p', 'h', '*', 'd', 'o', 's', 'p', 'h', '*', 'd']
colors = ['b', 'g', 'r', 'm', 'k', 'y', 'b', 'g', 'r', 'm', 'k', 'y']

# some common labels
sigma_names = ['\sigma_{11}', '\sigma_{22}', '\sigma_{33}', '\sigma_{23}', '\sigma_{13}', '\sigma_{12}']
epsilon_names = ['\epsilon_{11}', '\epsilon_{22}', '\epsilon_{33}', '\epsilon_{23}', '\epsilon_{13}', '\epsilon_{12}']
# Routines for Voigt calculus
# input and output of operations are symmetric 3x3 matrices as vector of length 6
# containing the coefficients 11 22 33 23 13 12
class Voigt:
	# identity matrix as vector
	Id = np.array([1, 1, 1, 0, 0, 0], dtype=np.double)
	Id9 = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.double)
	# identity tensor on symmetric 3x3 matrices as 6x6 matrix, i.e. Voigt.dyad4(A, Id4) = A
	Id4 = np.diag([1, 1, 1, 0.5, 0.5, 0.5])
	# trace of matrix
	@staticmethod
	def trace(a):
		return (a[0]+a[1]+a[2])
	# dyadic product A:b between tensor (as 6x6) and sym. 3x3 matrix (as vector)
	@staticmethod
	def dyad4(a, b):
		if (np.all(np.array(b.shape) <= 6)):
			b2 = b.copy()
			b2[3:6] *= 2
			return np.tensordot(a, b2, axes=1)
		return np.tensordot(a, b, axes=1)
	# dyadic product a:b between two sym. 3x3 matrices (as vector)
	@staticmethod
	def dyad(a, b):
		return ((a[0]*b[0]+a[1]*b[1]+a[2]*b[2]) + 2*(a[3]*b[3]+a[4]*b[4]+a[5]*b[5]))
	# symmetric product A*E + E*A
	@staticmethod
	def sym_prod(a, b):
		return np.array([
			2*(a[0]*b[0] + a[5]*b[5] + a[4]*b[4]),
			2*(a[5]*b[5] + a[1]*b[1] + a[3]*b[3]),
			2*(a[4]*b[4] + a[3]*b[3] + a[2]*b[2]),
			a[5]*b[4] + a[1]*b[3] + a[3]*b[2] + b[5]*a[4] + b[1]*a[3] + b[3]*a[2],
			a[0]*b[4] + a[5]*b[3] + a[4]*b[2] + b[0]*a[4] + b[5]*a[3] + b[4]*a[2],
			a[0]*b[5] + a[5]*b[1] + a[4]*b[3] + b[0]*a[5] + b[5]*a[1] + b[4]*a[3],
		])
	# return symmetric matrix from vector
	@staticmethod
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
	# return vector from symmetric matrix
	@staticmethod
	def vec(a, sym=True):
		if (sym):
			return np.array([a[0,0], a[1,1], a[2,2], a[1,2], a[0,2], a[0,1]])
		return np.array([a[0,0], a[1,1], a[2,2], a[1,2], a[0,2], a[0,1], a[2,1], a[2,0], a[1,0]])
	# return Voigt matrix from symmetric tensor
	@staticmethod
	def mat_from_tensor(T):
		A = np.zeros((6, 6), dtype=np.double)
		i1 = [0, 1, 2, 1, 0, 0]
		i2 = [0, 1, 2, 2, 2, 1]
		for i in range(6):
			for j in range(6):
				A[i,j] = T[i1[i], i2[i], i1[j], i2[j]]
		return A
	# return Voigt 9x9 matrix from tensor with major symmetry
	@staticmethod
	def mat_from_tensor_9x9(T):
		A = np.zeros((9, 9), dtype=np.double)
		i1 = [0, 1, 2, 1, 0, 0, 2, 2, 1]
		i2 = [0, 1, 2, 2, 2, 1, 1, 0, 0]
		for i in range(9):
			for j in range(9):
				A[i,j] = T[i1[i], i2[i], i1[j], i2[j]]
		return A

	# return Frobenius norm of 3x3 tensor
	@staticmethod
	def norm(T):
		return np.sqrt(Voigt.dyad(T,T))


def inspect():
	fg_gui = __import__("fibergen_gui")
	app = fg_gui.App(sys.argv)
	app.window.runProject(fg)
	app.exec_()


