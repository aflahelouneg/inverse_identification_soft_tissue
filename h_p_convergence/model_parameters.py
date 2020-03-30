'''
Material parameters of Gent model
'''

BIMATERIALS = True

if BIMATERIALS:
    parameters = {
        'mu_keloid' : 0.05,
        'jm_keloid' : 0.2,
        'mu_healthy' : 0.016,
        'jm_healthy' : 0.4,
    }
else:
    parameters = {
        'mu_keloid' : 0.016,
        'jm_keloid' : 0.4,
        'mu_healthy' : 0.016,
        'jm_healthy' : 0.4,
    }

interpolation = {
    'element_degree' : 2,
    }
