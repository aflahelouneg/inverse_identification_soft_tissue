
I installed fenics and other required packages using conda like this:
$ conda create -c conda-forge -n <name_of_environment> fenics=2018 scipy sympy numpy matplotlib

After you activate the environment, e.g.
$ source activate <name_of_environment>

You should be able to run an example problem as a module from the main directory like this:
$ python -m examples.basic_example_1
