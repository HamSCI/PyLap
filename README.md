## Install Instructions
1. Download PHaRLAP toolbox for matlab and unzip to a directory
2. Download Pylap 
3. Download Redistributable Libraries for Intel® C++ and Fortran 2020. Which is required because the original fortran code was compiled using the intel fortran compiler. The one used originally is availible at the following download link: https://registrationcenter-download.intel.com/akdlm/irc_nas/17113/l_comp_lib_2020.4.304_comp.for_redist.tgz
4. cd into the intel libraries folder and run install.sh, follow the prompt until install complete
```
4. export PHARLAP_HOME="your path to pharlap install dir"
5. export PYTHONPATH="Pylap_install_dir"
6. export LD_LIBRARY="/<YOUR PATH TO DIR>/l_comp_lib_2020.4.304_comp.for_redist/compilers_and_libraries_2020.4.304/linux/compiler/lib/intel64_lin" 
7. export DIR_MODELS_REF_DAT="{PHARLAP_HOME}/dat"

8. sudo apt-get install python3-tk python3-pil python3-pil.imagetk libqt5gui5 python3-pyqt5 
9. sudo apt-get install libxcb-randr0-dev libxcb-xtest0-dev libxcb-xinerama0-dev libxcb-shape0-dev libxcb-xkb-dev
10. source /home/{username}/bin/compilervars.sh intel64
11. sudo apt-get install python3-pip
12. pip3 install matplotlib numpy scipy qtpy
13. python3 setup.py install --user
```
14. Use Example folder files as templates to test the installation


