import subprocess
import generate_namelist


def main():
    schemes = [2, 4,  6]
    for scheme in schemes:
        nml = generate_namelist.DYCOMS_RF01()
        nml['meta']['simname'] = nml['meta'][
            'simname'] + '_order_' + str(scheme)
        nml['scalar_transport']['order'] = scheme
        nml['momentum_transport']['order'] = scheme
        nml['mpi']['nprocx'] = 4
        nml['mpi']['nprocy'] = 6
        nml['sgs']['scheme'] = 'UniformViscosity'
        nml['sgs']['UniformViscosity'] ={}
        nml['sgs']['UniformViscosity']['viscosity'] = 2.5


        generate_namelist.write_file(nml)
        run_str = 'bsub -n 24 mpirun python main.py ' + \
            nml['meta']['simname'] + '.in'
        print(run_str)
        subprocess.call([run_str], shell=True)

    return

if __name__ == "__main__":
    main()
