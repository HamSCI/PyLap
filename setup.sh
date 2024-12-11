
# check if the program is already setup

if [[ ${PHARLAP_HOME} = "" ]]; then


echo "First Time Setup"
echo "Please enter the filepath of the project"
echo "Example - /home/devindiehl/Pylap_project"
read x
echo ${x}

export PHARLAP_HOME="${x}/pharlap_4.5.1"
echo "export PHARLAP_HOME=${x}/pharlap_4.5.1" >> ~/.bashrc

if [[ -d "${x}/PyLap" ]] ;
then
export PYTHONPATH="${x}/PyLap" 
echo "export PYTHONPATH=${x}/PyLap" >> ~/.bashrc
echo "Pylap filepath found"
elif [[ -d "${x}/PyLap-main" ]] ;
then
export PYTHONPATH="${x}/PyLap-main" 
echo "export PYTHONPATH=${x}/PyLap-main" >> ~/.bashrc
echo "Pylap filepath found"
else 
echo "Pylap filepath not found"
fi

export LD_LIBRARY="${x}/l_comp_lib_2020.4.304_comp.for_redist/compilers_and_libraries_2020.4.304/linux/compiler/lib/intel64_lin"
echo "export LD_LIBRARY=${x}/l_comp_lib_2020.4.304_comp.for_redist/compilers_and_libraries_2020.4.304/linux/compiler/lib/intel64_lin" >> ~/.bashrc

export DIR_MODELS_REF_DAT="${x}/pharlap_4.5.1/dat"
echo "export DIR_MODELS_REF_DAT=${x}/pharlap_4.5.1/dat">> ~/.bashrc

source /home/$USER/bin/compilervars.sh intel64
echo "source /home/$USER/bin/compilervars.sh intel64" >>  ~/.bashrc


# echo "python3 setup.py install --user" >> ~/.bashrc

# export PHARLAP_HOME="/home/devindiehl/Pylap_project/PHARLAP/pharlap_4.5.1"

# export PYTHONPATH="/home/devindiehl/Pylap_project/PyLap"

# export LD_LIBRARY="/home/devindiehl/Pylap_project/l_comp_lib_2020.4.304_comp.for_redist/compilers_and_libraries_2020.4.304/linux/compiler/lib/intel64_lin" 

# export DIR_MODELS_REF_DAT="/home/devindiehl/Pylap_project/PHARLAP/pharlap_4.5.1/dat"


sudo apt-get install python3-tk python3-pil python3-pil.imagetk libqt5gui5 python3-pyqt5 

sudo apt-get install libxcb-randr0-dev libxcb-xtest0-dev libxcb-xinerama0-dev libxcb-shape0-dev libxcb-xkb-dev

sudo apt-get install python3-pip

# Create a venv and install the appropriate versions of numpy etc
python -m venv ${PYTHONPATH}
source ${PYTHONPATH}/bin/activate

pip3 install matplotlib==3.8.4 numpy==1.26.0 scipy==1.13.1 qtpy pyqt5

python3 setup.py install
else
echo "Pylap is already setup"
echo "if you wish to redo the setup enter the command 'nano .bashrc' in the home directory, delete the filepaths and then run the setup.sh script again"
fi
