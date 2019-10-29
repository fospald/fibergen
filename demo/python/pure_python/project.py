
# title: Pure Python
# description: Shows how to use fibergen in Python.

import fibergen

# create solver instance
fg = fibergen.FG()

# load project XML
# alternative: fg.load_xml("project.xml")
fg.set_xml(r'''
<settings>
	<title>Title</title>
	<solver n="16">
		<materials>
			<matrix E="1" nu="0.3" />
			<fiber  E="2" nu="0.3" />
		</materials>
	</solver>
	<actions>
		<select_material name="fiber" />
		<place_fiber R="0.5" />
		<run_load_case e11="1" />
		<run_load_case e22="1" />
		<python>
			print('variable is', variable)
		</python>
	</actions>
</settings>
''')

# modify settings
fg.set('solver..n', 32)
fg.set('solver.tol', 1e-8)
fg.set('title', 'New Title')
fg.set('solver.materials.fiber.', E=10, nu=0.35)
fg.set('actions.run_load_case[0].', e11=2)
fg.set('actions.run_load_case[1].', e22=0, e33=1)

# add convergence callback
def convergence_callback():
	global fg
	res = fg.get_residuals()[-1]
	print("convergence_callback called res =", res)
	return res < 1e-4
fg.set_convergence_callback(convergence_callback)

# set a python variable for the python context
fg.set_variable("variable", [1,2,3])

# print settings
print(fg.get_xml())

# run solver
fg.run()

# print phase names
phases = fg.get_phase_names()
print('phases:', phases)

# get volume fractions
for p in phases:
	print("volume fraction of %s = %g" % (p, fg.get_volume_fraction(p)))

# get displacement
u = fg.get_field('u')
print("shape of u:", u.shape)

# get mean stress
mean_stress = fg.get_mean_stress()
print("mean stress:", mean_stress)

