import subprocess
import generate_namelist


def main():
    schemes = [2, 3, 4, 5, 6, 7, 8, 9, 11]
    for scheme in schemes:
        nml = generate_namelist.StableBubble()
        nml['meta']['simname'] = nml['meta'][
            'simname'] + '_order_' + str(scheme)
        nml['scalar_transport']['order'] = scheme
        nml['momentum_transport']['order'] = scheme
        nml['mpi']['nprocx'] = 16

        generate_namelist.write_file(nml)
        run_str = 'bsub -n 16 mpirun python main.py ' + \
            nml['meta']['simname'] + '.in'
        print(run_str)
        subprocess.call([run_str], shell=True)

    return

if __name__ == "__main__":
    main()
