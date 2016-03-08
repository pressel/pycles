import argparse
import json
import numpy as np


def main():
    print('hello')

    # Parse information from the command line
    parser = argparse.ArgumentParser(prog='PyCLES')
    parser.add_argument("namelist")
    args = parser.parse_args()

    file_namelist = open(args.namelist).read()
    namelist = json.loads(file_namelist)
    del file_namelist



    a = np.linspace(0,99,100)
    print(a[0:5])

    k = 2
    m = 10
    b = np.ones((k,m))

    import TestRun
    Arr = TestRun.TestRun(namelist)
    a = Arr.array()



    if namelist['grid']['dims'] == 3:
        main3d(namelist)

    return


def main3d(namelist):
    print('calling main3d')

    # import TestArray_c
    # import Simulation3d
    import TestRun

    # test = TestArray_c.TestArray(namelist)
    # Simulation = Simulation3d.Simulation3d(namelist)
    test = TestRun.TestRun(namelist)

    #Simulation.initialize(namelist)
    test.initialize(namelist)

    #Simulation.run()
    # test.array_c()
    test.array()
    test.array_mean(namelist)
    test.hor_mean(namelist)


    return


if __name__ == "__main__":
    main()
