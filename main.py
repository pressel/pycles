import argparse
import json


def main():

    # Parse information from the command line
    parser = argparse.ArgumentParser(prog='PyCLES')
    parser.add_argument("namelist")
    args = parser.parse_args()

    file_namelist = open(args.namelist).read()
    namelist = json.loads(file_namelist)
    del file_namelist

    if namelist['grid']['dims'] == 3:
        main3d(namelist)

    return


def main3d(namelist):
    import Simulation3d

    Simulation = Simulation3d.Simulation3d(namelist)
    Simulation.initialize(namelist)
    Simulation.run()

    return


if __name__ == "__main__":
    main()
