# Install and run pycles

Instructions for conda on ch4 at IAC-ETHZ by Stefan Ruedisuehli (2023-01).

```bash
git clone git@github.com:ruestefa/pycles.git -b py3-conda
cd pycles
mamba env create -f environment.yml  # or requirements.yml
conda activate pycles
python generate_parameters.py
python setup.py build_ext --inplace
python generate_namelist.py StableBubble
python main.py StableBubble.in
```
