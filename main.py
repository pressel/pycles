import argparse
import pickle
def main():

    #Parse information from the command line
    parser = argparse.ArgumentParser(prog='PyCLES')
    parser.add_argument("namelist")
    parser.parse_args()
    args = parser.parse_args()

    file_namelist = open(args.namelist,'rb')
    namelist = pickle.load(file_namelist)
    file_namelist.close()
    del file_namelist



    if namelist['grid']['dims'] == 3:
        main3d(namelist)

    return



def main3d(namelist):
    import Simulation3d

    Simulation = Simulation3d.Simulation3d(namelist)

    return


if __name__ == "__main__":
    main()