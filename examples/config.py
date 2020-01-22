'''Configure stdout printing formats.'''

import sys
import numpy
import dolfin
import logging
import matplotlib


matplotlib.rc('lines',  linewidth=2,
                        markerfacecolor='w',
                        markeredgewidth=2.0,
                        markersize=8.0)
matplotlib.rc('axes',   titlesize='large',   # fontsize of the axes title
                        labelsize='large')   # fontsize of the x and y labels
matplotlib.rc('xtick',  labelsize='large')   # fontsize of the tick labels
matplotlib.rc('ytick',  labelsize='large')   # fontsize of the tick labels
matplotlib.rc('legend', fontsize='large')    # legend fontsize


numpy.set_printoptions(
    edgeitems = 4,
    threshold = 100,
    formatter = {'float' : '{: 13.6e}'.format},
    linewidth = 160)

logging.basicConfig(
    level=logging.INFO,
    format='\n%(levelname)s - %(funcName)s - %(filename)s:%(lineno)d\n'
           '  %(message)s\n',
    stream=sys.stdout)

logger = logging.getLogger()

dolfin.set_log_level(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
logging.getLogger('FFC').setLevel(logging.WARNING)
