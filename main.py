import argparse
import pickle
def main():

    #Parse information from the command line
    parser = argparse.ArgumentParser(prog='PyCLES')
    parser.add_argument("input")
    parser.add_argument("namelist")
    parser.parse_args()
    args = parser.parse_args()

    file_namelist = open(args.namelist,'rb')
    namelist = pickle.load(file_namelist)
    file_namelist.close()
    del file_namelist

    file_input = open(args.input,'rb')
    input = pickle.load(file_input)
    file_input.close()
    del file_input




    if namelist['grid']['dims'] == 3:
        main3d(input,namelist)





    return



def main3d(input,namelist):
    import Simulation3d

    Simulation = Simulation3d.Simulation3d(namelist)




    return




if __name__ == "__main__":
    main()