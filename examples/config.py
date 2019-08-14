'''Configure stdout printing formats.'''

import sys
import numpy
import dolfin
import logging

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
