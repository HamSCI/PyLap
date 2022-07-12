

export PHARLAP_HOME="/home/devindiehl/Pylap_project/PHARLAP/pharlap_4.5.0"

export PYTHONPATH="/home/devindiehl/Pylap_project/PyLap"

export LD_LIBRARY="/home/devindiehl/Pylap_project/l_comp_lib_2020.4.304_comp.for_redist/compilers_and_libraries_2020.4.304/linux/compiler/lib/intel64_lin" 

export DIR_MODELS_REF_DAT="/home/devindiehl/Pylap_project/PHARLAP/pharlap_4.5.0/dat"

sudo apt-get install python3-tk python3-pil python3-pil.imagetk libqt5gui5 python3-pyqt5 

sudo apt-get install libxcb-randr0-dev libxcb-xtest0-dev libxcb-xinerama0-dev libxcb-shape0-dev libxcb-xkb-dev

source /home/devindiehl/bin/compilervars.sh intel64

sudo apt-get install python3-pip

pip3 install matplotlib numpy scipy qtpy

python3 setup.py install --user
