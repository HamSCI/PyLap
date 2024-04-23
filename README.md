# Experimental Software Warning

PyLap is based on the scientific software PHaRLAP and is not meant for operational use as is stated on PHaRLAP's website. any and all risk falls to you if you are using this software.

## Linux Install Instructions ##

*Note* This install is only available for Ubuntu linux systems with X86 CPU's *Note* 

*Note*  It is also possible to install PyLap using WSL2 with Ubuntu for windows machines running windows 10 and windows 11 *Note* 

Download Steps

1. Create a folder on the home directory where you will place all of the downloaded resources.

2. Download PHaRLAP toolbox for matlab and unzip to the directory that you just created https://www.dst.defence.gov.au/our-technologies/pharlap-provision-high-frequency-raytracing-laboratory-propagation-studies .

3. Download Pylap from github in the same directory that you just created. 

    *Note* This works for either cloning the github repository or downloading it as a zip file *Note*

4. Download Redistributable Libraries for Intel® C++ and Fortran 2020. Which is required because the original fortran code was compiled using the Intel fortran compiler. The one used originally is available at the following download link: https://registrationcenter-download.intel.com/akdlm/irc_nas/17113/l_comp_lib_2020.4.304_comp.for_redist.tgz 

5. cd within a terminal window into the intel libraries folder and run install.sh, follow the prompt until install complete. when promted with where to install this, the default should be the home drive (ex. [/home/UserName]). The default location is fine to use for the installation.


File Directory model

├── PyLap Project folder  
   &emsp;├── PyLap\
   &emsp;├── PHARLAP\
   &emsp;└── l_comp_lib_2020.4.304_comp.for_redist\


## Setup with script file

1. cd into the pylap project directory.

2. Run the setup.sh script using the command “. ./setup.sh”. running this script will promt you to enter the filepath of the folder that all of your project is installed.

3. everything should be setup correctly! run the example files that are in the examples folder to make sure everything is setup correctly! if for some reason the example files do not work check the bashrc file in your home directory by using the command "nano .bashrc"(must be in the home directory).
  
    *Note* This is a one time setup and does not need to be run again unless a new install is made *Note*



## Manual setup

5. export PHARLAP_HOME="your path to pharlap install dir"

6. export PYTHONPATH="Pylap_install_dir"

7. export LD_LIBRARY="/"YOUR PATH TO DIR"/l_comp_lib_2020.4.304_comp.for_redist/compilers_and_libraries_2020.4.304/linux/compiler/lib/intel64_lin" 

8. export DIR_MODELS_REF_DAT="{PHARLAP_HOME}/dat"

9. sudo apt-get install python3-tk python3-pil python3-pil.imagetk libqt5gui5 python3-pyqt5 

10. sudo apt-get install libxcb-randr0-dev libxcb-xtest0-dev libxcb-xinerama0-dev libxcb-shape0-dev libxcb-xkb-dev

11. source /home/{username}/bin/compilervars.sh intel64

12. sudo apt-get install python3-pip

13. pip3 install matplotlib numpy scipy qtpy

14. cd $PYTHONPATH 

15. python3 setup.py install --user

16. Use Example folder files as templates to test the installation

    *Note* This is not a one time setup and will have to be redone if the terminal is closed out or if the code project is closed out *Note*



## for any questions or for help troubleshooting the install of PyLap please email Devin.diehl@scranton.edu ##
