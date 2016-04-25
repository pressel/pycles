import argparse
import json
import numpy as np


def main():

    # Parse information from the command line
    parser = argparse.ArgumentParser(prog='PyCLES')
    parser.add_argument("namelist")
    args = parser.parse_args()

    file_namelist = open(args.namelist).read()
    namelist = json.loads(file_namelist)
    del file_namelist


    import TestRun
    Arr = TestRun.TestRun(namelist)
    a = Arr.array()



    if namelist['grid']['dims'] == 3:
        main3d(namelist)

    return


def main3d(namelist):
    print('calling main3d')

    import TestRun

    test = TestRun.TestRun(namelist)

    test.initialize(namelist)

    test.array()
    test.array_mean(namelist)
    test.hor_mean(namelist)

    return


if __name__ == "__main__":
    main()
