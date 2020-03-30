'''
problems/healthy_skin_fixed_pads/mesh.py
'''

import os
import sys
import dolfin
import numpy as np
import matplotlib.pyplot as plt


MESHFILE_DIR = 'meshfiles/healthy_skin/pads_fixed'
MESHFILE_DOMAIN = 'mesh_healthy_skin_pads_fixed.xml'
MESHFILE_DOMAIN_ZOI = 'DIC_ZOI.xml'
MESHFILE_DOMAIN_MARKERS = 'mesh_healthy_skin_pads_fixed_physical_region.xml'
MESHFILE_BOUNDARY_MARKERS = 'mesh_healthy_skin_pads_fixed_facet_region.xml'


file_mesh_domain = os.path.join(os.curdir, MESHFILE_DIR, MESHFILE_DOMAIN)
file_mesh_domain_ZOI = os.path.join(os.curdir, MESHFILE_DIR, MESHFILE_DOMAIN_ZOI)
file_markers_domain = os.path.join(os.curdir, MESHFILE_DIR, MESHFILE_DOMAIN_MARKERS)
file_markers_boundary = os.path.join(os.curdir, MESHFILE_DIR, MESHFILE_BOUNDARY_MARKERS)

mesh_domain = dolfin.Mesh(file_mesh_domain)
mesh_domain_ZOI = dolfin.Mesh(file_mesh_domain_ZOI)
markers_domain = dolfin.MeshFunction('size_t', mesh_domain, file_markers_domain)
markers_boundary = dolfin.MeshFunction('size_t', mesh_domain, file_markers_boundary)


id_markers_domain = {
	'unmarked':        0,
	'keloid_measure' : 1,
	'healthy_measure': 2,
	'healthy':         3,
	}
id_markers_boundary = {
	'unmarked':  			0,
	'pad_one_external':		4,
	'pad_one_lateral':   	6,
	'pad_one_internal_left': 	7,
	'pad_one_internal_right': 	8,
	'pad_two':   5,
	}


if __name__ == '__mane__' and False:

    figname = "Mesh for the healthy Skin Model"
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
