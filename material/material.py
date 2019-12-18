'''
Some material models.

'''

import logging
import dolfin

from dolfin import Constant
from dolfin import Function
from dolfin import Identity
from dolfin import variable

from dolfin import det
from dolfin import diff
from dolfin import dot
from dolfin import exp
from dolfin import grad
from dolfin import inv
from dolfin import ln
from dolfin import tr

SEQUENCE_TYPES = (tuple, list)
logger = logging.getLogger()


class DeformationMeasures:
    def __init__(self, u):
        '''Deformation measures.'''

        # if not isinstance(u, Function):
        #     raise TypeError('Parameter `u` must be a `dolfin.Function`.')

        self.d = d = len(u)
        self.I = I = Identity(d)
        self.F = F = variable(I + grad(u))

        self.B = B = F*F.T
        self.C = C = F.T*F

        self.E = 0.5*(C-I)
        self.J = det(F)

        self.I1 = tr(C)
        self.I2 = 0.5*(tr(C)**2 - tr(C*C))
        self.I3 = det(C)


class MaterialModelBase:

    __ERROR_MESSAGE_MATERIAL_PARAMETERS = '`material_parameters` must be a `dict` ' \
                                          'or a `list` or `tuple` of `dict`s.'

    def __init__(self, material_parameters):
        '''Base class for deriving a specific material model.'''

        if isinstance(material_parameters, SEQUENCE_TYPES):

            if not all(isinstance(m, dict) for m in material_parameters):
                raise TypeError(self.__ERROR_MESSAGE_MATERIAL_PARAMETERS)

            self._material_parameters = \
                tuple({k: Constant(v) if isinstance(v, (float, int)) else v
                       for k, v in m.items()} for m in material_parameters)

            self._is_return_type_sequence = True

        else:

            if not isinstance(material_parameters, dict):
                raise TypeError(self.__ERROR_MESSAGE_MATERIAL_PARAMETERS)

            self._material_parameters = \
                ({k: Constant(v) if isinstance(v, (float, int))
                 else v for k, v in material_parameters.items()},)

            self._is_return_type_sequence = False

        # To be set during finalization
        self.deformation_measures = None

        self.psi = [] # Strain energy density
        self.pk1 = [] # First Piola-Kirchhoff stress
        self.pk2 = [] # Second Piola-Kirchhoff stress

    def initialize(self, u):
        '''To be extended by derived class.'''

        if self.is_initialized():
            logger.info('Re-initializing material model.')

            self.psi.clear()
            self.pk1.clear()
            self.pk2.clear()

        self.deformation_measures = DeformationMeasures(u)

        return self

    def is_initialized(self):
        '''Check if model has been initialized.'''
        return self.deformation_measures is not None and \
            bool(self.psi) and bool(self.pk1) and bool(self.pk2)

    def strain_energy_density(self):
        '''Material model strain energy density.'''
        if not self.is_initialized(): raise RuntimeError('Not initialized.')
        return self.psi if self._is_return_type_sequence else self.psi[0]

    def stress_measure_pk1(self):
        '''Material model First Piola-Kirchhoff stress measure.'''
        if not self.is_initialized(): raise RuntimeError('Not initialized.')
        return self.pk1 if self._is_return_type_sequence else self.pk1[0]

    def stress_measure_pk2(self):
        '''Material model Second Piola-Kirchhoff stress measure.'''
        if not self.is_initialized(): raise RuntimeError('Not initialized.')
        return self.pk2 if self._is_return_type_sequence else self.pk2[0]


class NeoHookean(MaterialModelBase):
    def initialize(self, u):
        super().initialize(u)

        d  = self.deformation_measures.d
        F  = self.deformation_measures.F
        J  = self.deformation_measures.J
        I1 = self.deformation_measures.I1
        # I2 = self.deformation_measures.I2
        # I3 = self.deformation_measures.I3

        for m in self._material_parameters:

            E  = m.get('E',  None)
            nu = m.get('nu', None)

            mu = m.get('mu', None)
            lm = m.get('lm', None)

            if mu is None:
                if E is None or nu is None:
                    raise RuntimeError('Material model requires parameter "mu"; '
                                       'otherwise, require parameters "E" and "nu".')

                mu = E/(2.0 + 2.0*nu)

            if lm is None:
                if E is None or nu is None:
                    raise RuntimeError('Material model requires parameter "lm"; '
                                       'otherwise, require parameters "E" and "nu".')

                lm = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))

            psi = (mu/2.0) * (I1 - d - 2.0*ln(J)) + (lm/2.0) * ln(J) ** 2

            pk1 = diff(psi, F)
            pk2 = dot(inv(F), pk1)

            self.psi.append(psi)
            self.pk1.append(pk1)
            self.pk2.append(pk2)

        return self


class StVenantKirchhoff(MaterialModelBase):
    def initialize(self, u):
        super().initialize(u)

        d  = self.deformation_measures.d
        F  = self.deformation_measures.F
        E  = self.deformation_measures.E

        for m in self._material_parameters:

            mu = m.get('mu', 0)
            lm = m.get('lm', 0)

            psi = 0

            if mu: psi += mu*tr(E*E)
            if lm: psi += lm/2.0*(tr(E)**2)

            pk1 = diff(psi, F)
            pk2 = dot(inv(F), pk1)

            self.psi.append(psi)
            self.pk1.append(pk1)
            self.pk2.append(pk2)

        return self


# class Yeoh(MaterialModelBase):
#     def initialize(self, u):
#         super().initialize(u)
#
#         d  = self.deformation_measures.d
#         F  = self.deformation_measures.F
#         I1 = self.deformation_measures.I1
#
#         for m in self._material_parameters:
#
#             expected_material_parameter_names = \
#                 [f'C{i}' for i in range(1, len(m)+1)]
#
#             if any(key_i not in expected_material_parameter_names for key_i in m.keys()):
#                 raise KeyError('Expected material parameter names for "Yeoh" '
#                                f'model: f{expected_material_parameter_names}')
#
#             Cs = (m[key_i] for key_i in expected_material_parameter_names)
#             psi = sum(C_i*(I1-d)**i for i, C_i in enumerate(Cs, start=1))
#
#             pk1 = diff(psi, F)
#             pk2 = dot(inv(F), pk1)
#
#             self.psi.append(psi)
#             self.pk1.append(pk1)
#             self.pk2.append(pk2)
#
#         return self


# class MooneyRivlin(MaterialModelBase):
#     def initialize(self, u):
#         super().initialize(u)
#
#         d  = self.deformation_measures.d
#         F  = self.deformation_measures.F
#         J  = self.deformation_measures.J
#         I1 = self.deformation_measures.I1
#         I2 = self.deformation_measures.I2
#         # I3 = self.deformation_measures.I3
#
#         # bar_I1 = I1 / J     if d == 2 else I1 / J**(2/d)
#         # bar_I2 = I2 / J**-2 if d == 2 else I2 / J**(4/d)
#
#         for m in self._material_parameters:
#
#             C1 = m.get('C1', 0)
#             C2 = m.get('C2', 0)
#             D1 = m.get('D1', 0)
#
#             # NOTE: In the limit of small strains
#             # mu = 2 * (C2 + C1); kappa = 2 * D1
#
#             psi = 0
#
#             if C1: psi += C1*(I1 - d)
#             if C2: psi += C2*(I2 - (d-1.0)*d/2.0)
#             if D1: psi += D1*(J - 1.0)**2
#
#             pk1 = diff(psi, F)
#             pk2 = dot(inv(F), pk1)
#
#             self.psi.append(psi)
#             self.pk1.append(pk1)
#             self.pk2.append(pk2)
#
#         return self
