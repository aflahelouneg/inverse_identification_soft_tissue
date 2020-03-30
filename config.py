
import sys
import numpy
import logging

numpy.set_printoptions(
    edgeitems = 4,
    threshold = 100,
    formatter = {'float' : '{: 13.6e}'.format},
    linewidth = 160)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.WARNING,
    format='\nLOGGER: %(funcName)s - %(levelname)s\n'
           '  %(message)s\n')
