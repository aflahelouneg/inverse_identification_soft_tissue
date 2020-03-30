import sys
import math
import logging
import importlib
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from dolfin import *
import dolfin
import time


# =============================================================
# *************************************************************
# =============================================================

PLOT_CONVERGENCE = True
PLOT_COSTS = True

# ------------------------------------------------------------------

number_of_elements = []
computation_cost_time_Lagrange1 = []
computation_cost_iteration_Lagrange1 = []

computation_cost_time_Lagrange2 = []
computation_cost_iteration_Lagrange2 = []

# ------------------------------------------------------------------
# ------------------------------------------------------------------

# Lagrange 1
# ----------
INTERPOLATION_DEGREE = 1 # Interpolation Lagrange 1

# ------------------------------------------------------------------

fin = open("h_p_convergence/model_parameters.py", "rt")
data = fin.read()

data = data.replace("'element_degree' : 1",
                    "'element_degree' : " + str(INTERPOLATION_DEGREE))
data = data.replace("'element_degree' : 2",
                    "'element_degree' : " + str(INTERPOLATION_DEGREE))
fin.close()

fin = open("h_p_convergence/model_parameters.py", "wt")
fin.write(data)
fin.close()

# ------------------------------------------------------------------

print('FEM solving on mesh 540-elements (Lagrange1)...')

time_start = time.time()
import h_p_convergence.keloid_skin_fem_540 as fem_540
time_end = time.time()

number_of_elements.append(540)
computation_cost_time_Lagrange1.append(time_end - time_start)
computation_cost_iteration_Lagrange1.append(fem_540.out[
    'FEM_solver_iterations'])
# ------------------------------------------------------------------

print('FEM solving on mesh 830-elements (Lagrange1)...')

time_start = time.time()
import h_p_convergence.keloid_skin_fem_830 as fem_830
time_end = time.time()

number_of_elements.append(830)
computation_cost_time_Lagrange1.append(time_end - time_start)
computation_cost_iteration_Lagrange1.append(fem_830.out[
    'FEM_solver_iterations'])
# ------------------------------------------------------------------

print('FEM solving on mesh 1300-elements (Lagrange1)...')

time_start = time.time()
import h_p_convergence.keloid_skin_fem_1300 as fem_1300
time_end = time.time()

number_of_elements.append(1300)
computation_cost_time_Lagrange1.append(time_end - time_start)
computation_cost_iteration_Lagrange1.append(fem_1300.out[
    'FEM_solver_iterations'])
# ------------------------------------------------------------------

print('FEM solving on mesh 1900-elements (Lagrange1)...')

time_start = time.time()
import h_p_convergence.keloid_skin_fem_1900 as fem_1900
time_end = time.time()

number_of_elements.append(1900)
computation_cost_time_Lagrange1.append(time_end - time_start)
computation_cost_iteration_Lagrange1.append(fem_1900.out[
    'FEM_solver_iterations'])
# ------------------------------------------------------------------

print('FEM solving on mesh 3000-elements (Lagrange1)...')

time_start = time.time()
import h_p_convergence.keloid_skin_fem_3000 as fem_3000
time_end = time.time()

number_of_elements.append(3000)
computation_cost_time_Lagrange1.append(time_end - time_start)
computation_cost_iteration_Lagrange1.append(fem_3000.out[
    'FEM_solver_iterations'])
# ------------------------------------------------------------------

print('FEM solving on mesh 4000-elements (Lagrange1)...')

time_start = time.time()
import h_p_convergence.keloid_skin_fem_4000 as fem_4000
time_end = time.time()

number_of_elements.append(4000)
computation_cost_time_Lagrange1.append(time_end - time_start)
computation_cost_iteration_Lagrange1.append(fem_4000.out[
    'FEM_solver_iterations'])
# ------------------------------------------------------------------

print('FEM solving on mesh 6000-elements (Lagrange1)...')

time_start = time.time()
import h_p_convergence.keloid_skin_fem_6000 as fem_6000
time_end = time.time()

number_of_elements.append(6000)
computation_cost_time_Lagrange1.append(time_end - time_start)
computation_cost_iteration_Lagrange1.append(fem_6000.out[
    'FEM_solver_iterations'])
# ------------------------------------------------------------------

print('FEM solving on mesh 9000-elements (Lagrange1)...')

time_start = time.time()
import h_p_convergence.keloid_skin_fem_9000 as fem_9000
time_end = time.time()

number_of_elements.append(9000)
computation_cost_time_Lagrange1.append(time_end - time_start)
computation_cost_iteration_Lagrange1.append(fem_9000.out[
    'FEM_solver_iterations'])
# ------------------------------------------------------------------

print('FEM solving on mesh 12000-elements (Lagrange1)...')

time_start = time.time()
import h_p_convergence.keloid_skin_fem_12000 as fem_12000
time_end = time.time()

number_of_elements.append(12000)
computation_cost_time_Lagrange1.append(time_end - time_start)
computation_cost_iteration_Lagrange1.append(fem_12000.out[
    'FEM_solver_iterations'])
# ------------------------------------------------------------------

print('FEM solving on mesh 22000-elements (Lagrange1)...')

time_start = time.time()
import h_p_convergence.keloid_skin_fem_22000 as fem_22000
time_end = time.time()

number_of_elements.append(22000)
computation_cost_time_Lagrange1.append(time_end - time_start)
computation_cost_iteration_Lagrange1.append(fem_22000.out[
    'FEM_solver_iterations'])
# ------------------------------------------------------------------

print('FEM solving on mesh 44000-elements (Lagrange1)...')

time_start = time.time()
import h_p_convergence.keloid_skin_fem_44000 as fem_44000
time_end = time.time()

number_of_elements.append(44000)
computation_cost_time_Lagrange1.append(time_end - time_start)
computation_cost_iteration_Lagrange1.append(fem_44000.out[
    'FEM_solver_iterations'])
# ------------------------------------------------------------------

u_Lagrange1 = [
	fem_540.out['displacement'],
	fem_830.out['displacement'],
	fem_1300.out['displacement'],
	fem_1900.out['displacement'],
    fem_3000.out['displacement'],
    fem_4000.out['displacement'],
    fem_6000.out['displacement'],
    fem_9000.out['displacement'],
    fem_12000.out['displacement'],
    fem_22000.out['displacement'],
    fem_44000.out['displacement']]

f_Lagrange1 = [
	fem_540.out['reaction_force'],
	fem_830.out['reaction_force'],
	fem_1300.out['reaction_force'],
	fem_1900.out['reaction_force'],
    fem_3000.out['reaction_force'],
    fem_4000.out['reaction_force'],
    fem_6000.out['reaction_force'],
    fem_9000.out['reaction_force'],
    fem_12000.out['reaction_force'],
    fem_22000.out['reaction_force'],
    fem_44000.out['reaction_force']]

# ------------------------------------------------------------------
# ------------------------------------------------------------------

# Lagrange 2
# ----------
INTERPOLATION_DEGREE = 2 # Interpolation Lagrange 1

# ------------------------------------------------------------------

fin = open("h_p_convergence/model_parameters.py", "rt")
data = fin.read()

data = data.replace("'element_degree' : 1",
                    "'element_degree' : " + str(INTERPOLATION_DEGREE))
data = data.replace("'element_degree' : 2",
                    "'element_degree' : " + str(INTERPOLATION_DEGREE))
fin.close()

fin = open("h_p_convergence/model_parameters.py", "wt")
fin.write(data)
fin.close()

# ------------------------------------------------------------------

print('FEM solving on mesh 540-elements (Lagrange2)...')

time_start = time.time()
importlib.reload(fem_540)
time_end = time.time()

computation_cost_time_Lagrange2.append(time_end - time_start)
computation_cost_iteration_Lagrange2.append(fem_540.out[
    'FEM_solver_iterations'])
# ------------------------------------------------------------------

print('FEM solving on mesh 830-elements (Lagrange2)...')

time_start = time.time()
importlib.reload(fem_830)
time_end = time.time()

computation_cost_time_Lagrange2.append(time_end - time_start)
computation_cost_iteration_Lagrange2.append(fem_830.out[
    'FEM_solver_iterations'])
# ------------------------------------------------------------------

print('FEM solving on mesh 1300-elements (Lagrange2)...')

time_start = time.time()
importlib.reload(fem_1300)
time_end = time.time()

computation_cost_time_Lagrange2.append(time_end - time_start)
computation_cost_iteration_Lagrange2.append(fem_1300.out[
    'FEM_solver_iterations'])
# ------------------------------------------------------------------

print('FEM solving on mesh 1900-elements (Lagrange2)...')

time_start = time.time()
importlib.reload(fem_1900)
time_end = time.time()

computation_cost_time_Lagrange2.append(time_end - time_start)
computation_cost_iteration_Lagrange2.append(fem_1900.out[
    'FEM_solver_iterations'])
# ------------------------------------------------------------------

print('FEM solving on mesh 3000-elements (Lagrange2)...')

time_start = time.time()
importlib.reload(fem_3000)
time_end = time.time()

computation_cost_time_Lagrange2.append(time_end - time_start)
computation_cost_iteration_Lagrange2.append(fem_3000.out[
    'FEM_solver_iterations'])
# ------------------------------------------------------------------

print('FEM solving on mesh 4000-elements (Lagrange2)...')

time_start = time.time()
importlib.reload(fem_4000)
time_end = time.time()

computation_cost_time_Lagrange2.append(time_end - time_start)
computation_cost_iteration_Lagrange2.append(fem_4000.out[
    'FEM_solver_iterations'])
# ------------------------------------------------------------------

print('FEM solving on mesh 6000-elements (Lagrange2)...')

time_start = time.time()
importlib.reload(fem_6000)
time_end = time.time()

computation_cost_time_Lagrange2.append(time_end - time_start)
computation_cost_iteration_Lagrange2.append(fem_6000.out[
    'FEM_solver_iterations'])
# ------------------------------------------------------------------

print('FEM solving on mesh 9000-elements (Lagrange2)...')

time_start = time.time()
importlib.reload(fem_9000)
time_end = time.time()

computation_cost_time_Lagrange2.append(time_end - time_start)
computation_cost_iteration_Lagrange2.append(fem_9000.out[
    'FEM_solver_iterations'])
# ------------------------------------------------------------------

print('FEM solving on mesh 12000-elements (Lagrange2)...')

time_start = time.time()
importlib.reload(fem_12000)
time_end = time.time()

computation_cost_time_Lagrange2.append(time_end - time_start)
computation_cost_iteration_Lagrange2.append(fem_12000.out[
    'FEM_solver_iterations'])
# ------------------------------------------------------------------

print('FEM solving on mesh 22000-elements (Lagrange2)...')

time_start = time.time()
importlib.reload(fem_22000)
time_end = time.time()

computation_cost_time_Lagrange2.append(time_end - time_start)
computation_cost_iteration_Lagrange2.append(fem_22000.out[
    'FEM_solver_iterations'])
# ------------------------------------------------------------------

print('FEM solving on mesh 44000-elements (Lagrange2)...')

time_start = time.time()
importlib.reload(fem_44000)
time_end = time.time()

computation_cost_time_Lagrange2.append(time_end - time_start)
computation_cost_iteration_Lagrange2.append(fem_44000.out[
    'FEM_solver_iterations'])
# ------------------------------------------------------------------

u_Lagrange2 = [
	fem_540.out['displacement'],
	fem_830.out['displacement'],
	fem_1300.out['displacement'],
	fem_1900.out['displacement'],
    fem_3000.out['displacement'],
    fem_4000.out['displacement'],
    fem_6000.out['displacement'],
    fem_9000.out['displacement'],
    fem_12000.out['displacement'],
    fem_22000.out['displacement'],
    fem_44000.out['displacement']]

f_Lagrange2 = [
	fem_540.out['reaction_force'],
	fem_830.out['reaction_force'],
	fem_1300.out['reaction_force'],
	fem_1900.out['reaction_force'],
    fem_3000.out['reaction_force'],
    fem_4000.out['reaction_force'],
    fem_6000.out['reaction_force'],
    fem_9000.out['reaction_force'],
    fem_12000.out['reaction_force'],
    fem_22000.out['reaction_force'],
    fem_44000.out['reaction_force']]

# ========================================================================
# ========================================================================


# The integrated force over the pad calculated in the mesh
# "with inter-pad gap Refined Lagrange 2" taken as reference


### Get reference mesh for projection

u_ref = u_Lagrange2[-1]
f_ref = f_Lagrange2[-1]

from h_p_convergence.keloid_skin_mesh_44000 import (
    mesh_domain)

V = dolfin.VectorFunctionSpace(mesh_domain, 'Lagrange', 2)

errU_L1 = []
errF_L1 = []

errU_L2 = []
errF_L2 = []

# Exporting mismath solutions fields Lagrage 1
for k, h in enumerate(number_of_elements):

    print("Computing relative mismatches ", k+1, "/", len(number_of_elements))
    u_proj_L2 = dolfin.project(u_Lagrange1[k], V)

    diff_u = u_ref - u_proj_L2
    diff_u = dolfin.project(diff_u, V)

    errU_L1.append(dolfin.norm(diff_u, 'L2')/dolfin.norm(u_ref, 'L2'))
    errF_L1.append((f_Lagrange1[k] - f_ref)**2/f_ref**2)

    file = File("h_p_convergence/Paraview_fields/displacement_mismatch_Lagrange1_" +
                str(number_of_elements[k]) + ".pvd")
    file << diff_u

# Exporting mismath solutions fields Lagrage 2
for k, h in enumerate(number_of_elements):

    print("Computing relative mismatches ", k+1, "/", len(number_of_elements))
    u_proj_L2 = dolfin.project(u_Lagrange2[k], V)

    diff_u = u_ref - u_proj_L2
    diff_u = dolfin.project(diff_u, V)

    errU_L2.append(dolfin.norm(diff_u, 'L2')/dolfin.norm(u_ref, 'L2'))
    errF_L2.append((f_Lagrange2[k] - f_ref)**2/f_ref**2)

    file = File("h_p_convergence/Paraview_fields/displacement_mismatch_Lagrange2_" +
                str(number_of_elements[k]) + ".pvd")
    file << diff_u


## PLOT
if PLOT_CONVERGENCE:
    figname = 'Mismatch on displacement (bimaterial)'
    plt.figure(figname)
    plt.clf()

    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)

    plt.loglog(number_of_elements, errU_L1, 'b-*')
    plt.loglog(number_of_elements[0:-1], errU_L2[0:-1], 'r-*')

    plt.legend(['Lagrange 1', 'Lagrange 2'])

    plt.xlabel('Number of elements []')
    plt.ylabel('Relative error []')
    plt.title(figname)
    plt.savefig('h_p_convergence/plots/errU.eps')

if PLOT_CONVERGENCE:
    figname = 'Mismatch on reaction force (bimaterial)'
    plt.figure(figname)
    plt.clf()

    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)

    plt.loglog(number_of_elements, errF_L1, 'b-*')
    plt.loglog(number_of_elements[0:-1], errF_L2[0:-1], 'r-*')

    plt.legend(['Lagrange 1', 'Lagrange 2'])

    plt.xlabel('Number of elements []')
    plt.ylabel('Relative error []')
    plt.title(figname)
    plt.savefig('h_p_convergence/plots/errF.eps')


if PLOT_COSTS:
    figname = 'Forward FEM computational cost (over time)'
    plt.figure(figname)
    plt.clf()

    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)

    plt.loglog(number_of_elements, computation_cost_time_Lagrange2, 'r-*')

    plt.legend(['Lagrange 2'])

    plt.xlabel('Number of elements []')
    plt.ylabel('Computation time [s]')
    plt.title(figname)
    plt.savefig('h_p_convergence/plots/cost_time.eps')

if PLOT_COSTS:
    figname = 'Forward FEM computational cost (over number of iterations)'
    plt.figure(figname)
    plt.clf()

    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)

    plt.semilogx(number_of_elements, computation_cost_iteration_Lagrange2, 'r-*')

    plt.legend(['Lagrange 2'])

    plt.xlabel('Number of elements []')
    plt.ylabel('Number of iteration []')
    plt.title(figname)
    plt.savefig('h_p_convergence/plots/cost_interation.eps')


result_parameters_file = open('h_p_convergence_results.txt', 'w')

result_parameters_file.write('Mehses: ' + str(number_of_elements) + '\n')
result_parameters_file.write('ErrU Lagrange1: ' + str(errU_L1) + '\n')
result_parameters_file.write('ErrF Lagrange1: ' + str(errF_L1) +'\n')
result_parameters_file.write('ErrU Lagrange2: ' + str(errU_L2) + '\n')
result_parameters_file.write('ErrF Lagrange2: ' + str(errF_L2) + '\n')
result_parameters_file.write('computation cost time Lagrange2: '\
                            + str(computation_cost_time_Lagrange2) + '\n')
result_parameters_file.write('computation cost iteration Lagrange2: '\
                            + str(computation_cost_iteration_Lagrange2)+ '\n')

result_parameters_file.close()
