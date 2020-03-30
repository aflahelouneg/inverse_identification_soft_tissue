'''
measure/prepare.py

TODO:
  - data fitting
  - data evaluation/interpolation

'''

import os
import sys
import time

import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt


def weighted_average_filter(a, w, count=1,
    overwrite_a=False, overwrite_w=False):
    '''Weighted mean filter along first dimension.

    Returns
    -------
    a : numpy.ndarray (nD)
        The array after the filtering has been applied `count` times.
    w : numpy.ndarray (1D)

    '''
    a = np.array(a, float, ndmin=1, copy=not overwrite_a)
    w = np.array(w, float, ndmin=1, copy=not overwrite_w)

    if len(w) % 2 != 1:
        raise ValueError('Number of weights (`len(w)`) must be an odd number.')
		
    a_tmp = []
    w /= w.sum()
    i0 = (len(w)-1)//2

    for _ in range(count):
        for i in range(i0, len(a)-i0):
            a_tmp.append(w.dot(a[i-i0:i+i0+1]))
        a[i0:len(a)-i0] = a_tmp; a_tmp.clear()

    return a


class ForceMeasurement:
    def __init__(self, uk, fk):
        '''
        Parameters
        ----------
        uk : 1D or 2D array of floats
            Displacement vector (row) of a point at each time.
        fk : 1D or 2D array of floats
            Correspondin force vector (row) for each time.

        Returns
        -------
        None

        '''

        uk = np.asarray(uk, float)
        fk = np.asarray(fk, float)

        if uk.ndim > 2:
            raise TypeError('Expected `uk` to be array-like '
                'with a maxium dimension of 2.')

        if fk.ndim > 2:
            raise TypeError('Expected `fk` to be array-like '
                'with a maxium dimension of 2.')
				
        if uk.ndim == 1:
            uk = uk[:,None]
			
        elif uk.ndim == 0:
            uk = uk[None,None]

        if fk.ndim == 1:
            fk = fk[:,None]
        elif fk.ndim == 0:
            fk = fk[None,None]

        if len(uk) != len(fk):
            raise TypeError('Expected the same number of time points, i.e.'
                'the same size of the first dimension of `uk` and `fk`.')

        self.uk = uk
        self.fk = fk

    @classmethod
    def load_from_files(cls, filepath_uk, filepath_fk, delimiter=None):
        '''Load arrays of displacements and of the corresponding forces.'''

        uk = np.loadtxt(filepath_uk, dtype=float, delimiter=delimiter, ndmin=1)
        fk = np.loadtxt(filepath_fk, dtype=float, delimiter=delimiter, ndmin=1)
		
        if len(uk) != len(fk):
            raise TypeError('Expected the same number of time points, i.e.'
                'the same size of the first dimension of `uk` and `fk`.')

        return cls(uk, fk)

    @staticmethod
    def get_filtered_values(a, w=None, count=1):
        '''Weighted average filtering in the time dimension.'''

        if w is None: w = np.array((0.25,0.50,0.25))
        else: w = np.array(w, ndmin=1, copy=False)
        a = weighted_average_filter(a, w, count)

        return a

    def view_displacements(self):
        '''return copy'''
        return self.uk.copy()

    def view_forces(self):
        '''return copy'''
        return self.fk.copy()

    def rescale_forces(self, scale):
        self.fk *= scale

    def filter_displacements(self, w=None, count=1):
        '''Weighted average filtering in the time dimension.'''
        self.uk[:] = self.get_filtered_values(self.uk, w, count)

    def filter_forces(self, w=None, count=1):
        '''Weighted average filtering in the time dimension.'''
        self.fk[:] = self.get_filtered_values(self.fk, w, count)

    def remove_data_points(self, uk, fk, atol_u, atol_f, dryrun=False):
        '''Remove points that are within tolerance of specified values.'''

        atol_u **= 2
        atol_f **= 2

        mask_keep = np.ones(len(self.uk), dtype=bool)

        for uk_i, fk_i in zip(uk, fk):
            mask_match = np.logical_and(
                np.sum((self.uk-uk_i)**2, axis=1) < atol_u,
                np.sum((self.fk-fk_i)**2, axis=1) < atol_f)
            mask_keep[mask_match] = False

        mask_remove = np.logical_not(mask_keep)
        ids_remove = np.argwhere(mask_remove)

        uk_remove = self.uk[ids_remove].copy()
        fk_remove = self.fk[ids_remove].copy()

        if not dryrun and ids_remove:

            self.uk = self.uk[mask_keep]
            self.fk = self.fk[mask_keep]

            if not self.uk.flags.owndata:
                self.uk = np.array(self.uk)

            if not self.fk.flags.owndata:
                self.fk = np.array(self.fk)

        return uk_remove, fk_remove, ids_remove


class DisplacementMeasurement:
    def __init__(self, xk, uk, values_relative=False):

        if not isinstance(xk, np.ndarray):
            raise TypeError('Expected `xk` to be a `numpy.ndarray`.')

        if not isinstance(uk, list) or not all(isinstance(uk_t, np.ndarray) for uk_t in uk):
            raise TypeError('Expected `uk` to be a `list` of `numpy.ndarray`s.')

        if not all(xk.shape == uk_t.shape for uk_t in uk):
            raise TypeError('Expected items in `uk` to have the same shape as `xk`.')

        self.values_relative = bool(values_relative)

        self.xk = xk
        self.uk = uk

    @classmethod
    def load_from_files(cls,
        filepath_xk, filepath_yk, filepath_uk, filepath_vk,
        delimiter=None, values_relative=False):
        '''Firstly, load each position array (1D) and each (transient) value array
        (2D). Secondly, combine the 1D position arrays into a single 2D array and
        combine the 2D value arrays into a list of 2D arrays. Finally, return 2D
        position array and the list of 2D values arrays.'''

        xk = np.loadtxt(filepath_xk, dtype=float, delimiter=delimiter, ndmin=1)
        yk = np.loadtxt(filepath_yk, dtype=float, delimiter=delimiter, ndmin=1)
        uk = np.loadtxt(filepath_uk, dtype=float, delimiter=delimiter, ndmin=1)
        vk = np.loadtxt(filepath_vk, dtype=float, delimiter=delimiter, ndmin=1)

        if xk.shape != yk.shape:
            raise TypeError('Expected same shapes of `xk` and `yk`.')

        if uk.shape != vk.shape:
            raise TypeError('Expected same shapes of `uk` and `vk`.')
	
        # print(len(xk))
        # print(len(uk))
        # input("press to continue ...")		

        if len(xk) != len(uk):
            raise TypeError('Expected the same number of points, i.e. '
                'the same size of the first dimension of `xk` and `uk`.')

        if uk.ndim == 1:
            uk = uk[:,None] # new axis
            vk = vk[:,None]

        nt = uk.shape[1]
        # print(nt)
        # input("press to continue ...")
		
        uk = np.split(uk, nt, axis=1)
        vk = np.split(vk, nt, axis=1)

        xk = np.stack((xk, yk), axis=1)
        uk = [np.concatenate(uk_t, axis=1)
            for uk_t in zip(uk, vk)]

        return cls(xk, uk, values_relative)


    def __add__(self, other):

        cls = self.__class__

        if not isinstance(other, cls):
            raise TypeError

        if other.values_relative != self.values_relative:
            raise TypeError

        if other.xk.shape[1] != self.xk.shape[1]:
            raise TypeError

        if len(other.uk) != len(self.uk):
            raise TypeError

        xk = np.concatenate((self.xk, other.xk), axis=0)
        uk = [np.concatenate((self_uk_t, other_uk_t), axis=0)
            for self_uk_t, other_uk_t in zip(self.uk, other.uk)]

        return cls(xk, uk, self.values_relative)


    def set_values_relative(self):
        if not self.values_relative:
            self.values_relative = True
            for uk_t in self.uk:
                uk_t -= self.xk

    def set_values_total(self):
        if self.values_relative:
            self.values_relative = False
            for uk_t in self.uk:
                uk_t += self.xk


    def view_coords(self):
        '''return copy'''
        return self.xk.copy()

    def view_values(self):
        '''return copy'''
        return [uk_t.copy() for uk_t in self.uk]

    def view_displacements(self):
        uk = self.view_values()
        if not self.values_relative:
            for uk_t in uk:
                uk_t -= self.xk
        return uk

    def view_positions(self):
        uk = self.view_values()
        if self.values_relative:
            for uk_t in uk:
                uk_t += self.xk
        return uk


    def compute_mean_coords(self):
        '''Compute mean values for each time.'''
        return self.xk.mean(axis=0)

    def compute_mean_values(self):
        '''Compute mean values for each time.'''
        um = []
        for uk_t in self.uk:
            um.append(uk_t.mean(axis=0))
        return np.stack(um, axis=0)

    def compute_mean_displacements(self):
        '''Compute mean values for each time.'''
        um = []
        for uk_t in self.view_displacements():
            um.append(uk_t.mean(axis=0))
        return np.stack(um, axis=0)

    def compute_mean_positions(self):
        '''Compute mean values for each time.'''
        um = []
        for uk_t in self.view_positions():
            um.append(uk_t.mean(axis=0))
        return np.stack(um, axis=0)

    def compute_mean_rotations(self):
        '''Compute mean values for each time.'''

        uk = self.view_positions()
        um = self.compute_mean_positions()

        rk_t0 = (uk[0] - um[0]).T
        rk_t0 /= np.sqrt((rk_t0**2).sum(axis=0))

        th = [0.0] # incremental mean rotations
        for uk_t1, um_t1 in zip(uk[1:], um[1:]):

            rk_t1 = (uk_t1 - um_t1).T
            rk_t1 /= np.sqrt((rk_t1**2).sum(axis=0))

            s = np.cross(rk_t0, rk_t1, axis=0)
            c = np.sum(rk_t0*rk_t1, axis=0)
            th.append(np.arctan2(s,c).mean())

            rk_t0 = rk_t1

        return np.array(th).cumsum()


    def offset_coords(self, x, i=slice(0,2), operator='+'):

        if operator == '+':
            self.xk[:,i] += x
        elif operator == '-':
            self.xk[:,i] -= x
        else:
            raise TypeError('`operator`: "+" or "-" ?')

    def offset_values(self, u, i=slice(0,2), operator='+'):

        if len(u) != len(self.uk):
            raise TypeError('Number of time points.')

        if operator == '+':
            for self_uk_t, u_t in zip(self.uk, u):
                self_uk_t[:,i] = self_uk_t[:,i] + u_t[i]
        elif operator == '-':
            for self_uk_t, u_t in zip(self.uk, u):
                self_uk_t[:,i] = self_uk_t[:,i] - u_t[i]
        else:
            raise TypeError('`operator`: "+" or "-" ?')

    def _rotate_axis(self, th):

        raise NotImplementedError

        if not isinstance(th, (float,int)):
            th = float(th) # try any way

        c = np.cos(th); s = np.sin(th)
        RT = np.array([[c,s],[-s,c]])

        xk = self.xk
        uk = self.uk

        if self.values_relative:
            for uk_t in uk:
                uk_t *= uk_t.dot(RT)
            xk[:] = xk.dot(RT)
            for uk_t in uk:
                uk_t[:] -= xk
        else:
            for uk_t in uk:
                uk_t[:] = uk_t.dot(RT)

        self.xk[:] = self.xk.dot(RT)

        self.xk[:] = self.xk.dot(np.array([[c,s],[-s,c]]))

    def _rotate_coord_axis(self, th, x0=None):
        # the opposite of rotating value

        if not isinstance(th, (float,int)):
            th = float(th) # try any way

        if x0 is None:
            x0 = np.array([0.0,0.0])
        else:
            x0 = np.asarray(x0)

        c = np.cos(th)
        s = np.sin(th)

        self.xk[:] = x0 + (self.xk-x0).dot(np.array([[c,-s],[s,c]]))

    def _rotate_value_axis(self, th, x0):
        '''Rotate value axess by `th` angles. Or, in other words, remove
        the rotations of data values by `th`. `th` is possitive anticlock-wise.

        Parameters
        ----------
        `th` : array of floats
        The angles of axis rotation for each time-step that are to be applied.

        '''

        if not isinstance(th, np.ndarray):
            th = np.array(th, ndmin=1)

        if len(th) != len(self.uk):
            if len(th) == 1:
                th = np.full((len(self.uk),), th)
            else:
                raise TypeError

        c = np.cos(th)
        s = np.sin(th)

        nx = np.stack([ c, s], axis=1)
        ny = np.stack([-s, c], axis=1)

        xk = self.xk
        uk = self.uk

        if self.values_relative:
            for uk_t in uk: uk_t += xk

        for uk_t in uk:
            uk_t -= x0

        for uk_t, nx_t, ny_t in zip(uk, nx, ny):
            uk_t[:,0], uk_t[:,1] = \
                (uk_t*nx_t).sum(axis=1), (uk_t*ny_t).sum(axis=1)

        for uk_t in uk:
            uk_t += x0

        if self.values_relative:
            for uk_t in uk: uk_t -= xk


    def rescale(self, scale_x, scale_y=None):
        '''Rescale grid and values.'''

        if scale_y is None:
            scale_y = scale_x

        xk = self.xk
        uk = self.uk

        if not self.values_relative:
            for uk_t in uk: uk_t -= xk

        # xk -= xk.min(axis=0)
        xk[:,0] *= scale_x
        xk[:,1] *= scale_y
        # xk += x0

        for uk_t in uk:
            uk_t[:,0] *= scale_x
            uk_t[:,1] *= scale_y

        if not self.values_relative:
            for uk_t in uk: uk_t += xk

        x0 = xk.min(axis=0)
        x1 = xk.max(axis=0)

        return x0, x1

    def rescale_to_rectangle(self, p0, p1):
        '''Rescale grid and values to a rectangle.'''

        x0, y0, *_ = p0
        x1, y1, *_ = p1

        xk = self.xk
        uk = self.uk

        if not self.values_relative:
            for uk_t in uk: uk_t -= xk

        X_min, Y_min = xk.min(axis=0)
        X_max, Y_max = xk.max(axis=0)

        xk[:,0] -= X_min
        xk[:,1] -= Y_min

        scale_x = (x1-x0)/(X_max-X_min)
        scale_y = (y1-y0)/(Y_max-Y_min)

        xk[:,0] *= scale_x
        xk[:,1] *= scale_y

        xk[:,0] += x0
        xk[:,1] += y0

        for uk_t in uk:
            uk_t[:,0] *= scale_x
            uk_t[:,1] *= scale_y

        if not self.values_relative:
            for uk_t in uk: uk_t += xk

        return scale_x, scale_y


    def remove_data_points(self, xk, atol=None, dryrun=False):

        ids_remove = []
        for xk_i in xk:
            ids_remove.append(np.argmin(
                np.sum((self.xk-xk_i)**2, axis=1)))

        if atol:
            r2 = np.sum((self.xk[ids_remove]-xk)**2, axis=1)
            ids_remove = [idx_i for idx_i, r2_i in \
                zip(ids_remove, r2) if r2_i < atol**2]

        mask_keep = np.ones(len(self.xk), dtype=bool)
        mask_keep[ids_remove] = False

        xk_remove = self.xk[ids_remove].copy()
        uk_remove = [uk_t[ids_remove].copy() for uk_t in self.uk]

        if not dryrun and ids_remove:

            self.xk = self.xk[mask_keep]
            self.uk = [uk_t[mask_keep] for uk_t in self.uk]

            if not self.xk.flags.owndata:
                self.xk = np.array(self.xk)

            if not self.uk[0].flags.owndata:
                self.uk = [np.array(self.uk) for uk_t in self.uk]

        return xk_remove, uk_remove, ids_remove

    def filter_values(self, w=None, count=1):
        '''Weighted average filtering in the time dimension.'''

        if w is None: w = np.array((0.25,0.50,0.25))
        else: w = np.array(w, ndmin=1, copy=True)

        nt = len(self.uk)
        nk = len(self.uk[0])
        nd = len(self.uk[0][0])

        uk = np.array(self.uk, ndmin=3, copy=True).reshape((nt,nk*nd))
        uk = weighted_average_filter(uk, w, count, overwrite_a=True)
        uk = uk.reshape((nt,nk,nd))

        for self_uk_t, uk_t in zip(self.uk, uk):
            self_uk_t[:] = uk_t


    def plot_data(self, fig_name='', dt=0.25, index=None):

        fh = plt.figure(fig_name)
        fh.clear(); fh.show()

        ax = fh.add_subplot(1,1,1)

        xk = self.view_coords()
        uk = self.view_displacements()

        x_min = min(min(xk[:,0]+uk_t[:,0]) for uk_t in (uk[0], uk[-1]))
        x_max = max(max(xk[:,0]+uk_t[:,0]) for uk_t in (uk[0], uk[-1]))
        y_min = min(min(xk[:,1]+uk_t[:,1]) for uk_t in (uk[0], uk[-1]))
        y_max = max(max(xk[:,1]+uk_t[:,1]) for uk_t in (uk[0], uk[-1]))

        marg = max(x_max-x_min, y_max-y_min) * 0.1

        xlim = (x_min-marg, x_max+marg)
        ylim = (y_min-marg, y_max+marg)

        print(xlim)
        print(ylim)

        ck = [np.sqrt(np.sum(uk_tj**2 for uk_tj in uk_t.T)) for uk_t in uk]

        vmin = min(min(ck_t) for ck_t in ck)
        vmax = max(max(ck_t) for ck_t in ck)

        if index is not None:
            uk = (uk[index],)
            ck = (ck[index],)

        for uk_t, ck_t in zip(uk, ck):

            ax.clear()

            ax.scatter(xk[:,0]+uk_t[:,0], xk[:,1]+uk_t[:,1], c='w', s=10,
                marker='o', vmin=vmin, vmax=vmax, edgecolor='k', alpha=0.5)

            ax.scatter(xk[:,0], xk[:,1], c='r', s=5, marker='+')

            ax.axis('equal')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            ax.legend(['positions at time', 'initial positions'])

            ax.set_title(fig_name)
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')

            fh.canvas.draw()
            fh.canvas.flush_events()

            time.sleep(dt)
