'''
problems/healthy_skin_fixed_pads/mesh.py
'''

import os
import sys
import dolfin
import numpy as np
import matplotlib.pyplot as plt


MESHFILE_DIR = 'meshfiles/keloid_skin/pads_fixed_1300_cells'
MESHFILE_DOMAIN = 'mesh_keloid_skin_pads_fixed.xml'
MESHFILE_DOMAIN_MARKERS = 'mesh_keloid_skin_pads_fixed_physical_region.xml'
MESHFILE_BOUNDARY_MARKERS = 'mesh_keloid_skin_pads_fixed_facet_region.xml'


file_mesh_domain = os.path.join(os.curdir, MESHFILE_DIR, MESHFILE_DOMAIN)
file_markers_domain = os.path.join(os.curdir, MESHFILE_DIR, MESHFILE_DOMAIN_MARKERS)
file_markers_boundary = os.path.join(os.curdir, MESHFILE_DIR, MESHFILE_BOUNDARY_MARKERS)

mesh_domain = dolfin.Mesh(file_mesh_domain)
markers_domain = dolfin.MeshFunction('size_t', mesh_domain, file_markers_domain)
markers_boundary = dolfin.MeshFunction('size_t', mesh_domain, file_markers_boundary)
# print(markers_boundary)
# input("stop...")


id_markers_domain = {
	'unmarked':        0,
	'keloid_measure':  10,
	'healthy_measure': 20,
	'healthy':         30,
	}
id_markers_boundary = {
	'unmarked':  			0,
	'pad_one':   			1,
	'pad_one_sensor': 		3,
	'pad_two':     			2,
	}




if __name__ == '__mane__':

    figname = "Mesh for the Keloid Skin Model"
    fh = plt.figure(figname)
    fh.clear()

    df.plot(mesh_domain)
    df.plot(markers_domain)

    ax = fh.gca()
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')

    # ax.set_title('Awesome Mix Vol.1')
    ax.set_title(figname)
    plt.show()
