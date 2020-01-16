'''
Execute script of specific version
'''

import sys

AVAILABLE_VERSIONS_OF_MAIN = (1, 2)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        try:
            version_of_main = int(sys.argv[1])
        except:
            raise TypeError("Command-line argument for the version of "
                            "the main script must be an integer in "
                            f"{AVAILABLE_VERSIONS_OF_MAIN}")
    else:
        try:
            version_of_main = int(input("Input version of the main script: "))
        except:
            raise TypeError("The input argument for the version of "
                            "the main script must be an intger in "
                            f"{AVAILABLE_VERSIONS_OF_MAIN}")

    if version_of_main not in AVAILABLE_VERSIONS_OF_MAIN:
        raise ValueError("The version of the main script must be an "
                         f"integer in {AVAILABLE_VERSIONS_OF_MAIN}")

    if VERSION_OF_MAIN == 1:
        from .main_v1 import *
    elif VERSION_OF_MAIN == 2:
        from .main_v2 import *
    else:
        raise RuntimeError
