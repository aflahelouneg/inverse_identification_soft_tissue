"""

"""

import sys
import dolfin
import numpy as np
import os.path as path


PARENT_DIRECTORY = path.dirname(path.relpath(__file__))
MESHFILES_DIRECTORY = path.join(PARENT_DIRECTORY, 'meshfiles')

MESHFILE_DOMAIN           = 'mesh.xml'
MESHFILE_INTERIOR_MARKERS = 'mesh_physical_region.xml'
MESHFILE_BOUNDARY_MARKERS = 'mesh_facet_region.xml'


def load_mesh(subdir, pardir=MESHFILES_DIRECTORY):

    directory = path.join(pardir, subdir)

    mesh = dolfin.Mesh(path.join(directory, MESHFILE_DOMAIN))

    domain_markers = dolfin.MeshFunction('size_t', mesh,
        path.join(directory, MESHFILE_INTERIOR_MARKERS))

    boundary_markers = dolfin.MeshFunction('size_t', mesh,
        path.join(directory, MESHFILE_BOUNDARY_MARKERS))

    # NOTE: Subdomain id's will generally be a sequence of sequences.

    mesh_data = {
        'mesh':             mesh,
        'domain_markers':   domain_markers,
        'boundary_markers': boundary_markers,
        'id_subdomains_material': [(1,),],
        'id_subdomains_dic': None,
        'id_boundaries_pad_moving': [5,],
        'id_boundaries_pad_fixed': [6,],
        'id_boundaries_exterior': [1,2,3,4]
        }

    return mesh_data
