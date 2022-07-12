## Install Instructions

1.Download PHaRLAP toolbox for matlab and unzip to a directory

Note* the file structure should consist of one containing folder on the home to install the pylap, pharlap, and intel libraries to.

2.Download Pylap

3. Download Redistributable Libraries for Intel® C++ and Fortran 2020. Which is required because the original fortran code was compiled using the Intel fortran compiler. The one used originally is available at the following download link: https://registrationcenter-download.intel.com/akdlm/irc_nas/17113/l_comp_lib_2020.4.304_comp.for_redist.tgz

4. cd within a terminal window into the intel libraries folder and run install.sh, follow the prompt until install complete

Note* you must run the below commands from within the pylap project wether that means within the editor or a terminal that you have used cd to enter the install directory.
  
Also a setup.sh file has been created to run the below commands all at once. You may run this file within the terminal that the example files will also be run by running the command “. ./setup.sh”. Note that before you run this command the filepaths within the setup.sh file that is in the pylap project must be changed to fit your local filepaths. You can look at the current sh file to see how to properly set the filepaths.

5. export PHARLAP_HOME="your path to pharlap install dir"

6. export PYTHONPATH="Pylap_install_dir"

7. export LD_LIBRARY="/<YOUR PATH TO DIR>/l_comp_lib_2020.4.304_comp.for_redist/compilers_and_libraries_2020.4.304/linux/compiler/lib/intel64_lin" 

8. export DIR_MODELS_REF_DAT="{PHARLAP_HOME}/dat"

9. sudo apt-get install python3-tk python3-pil python3-pil.imagetk libqt5gui5 python3-pyqt5 

10. sudo apt-get install libxcb-randr0-dev libxcb-xtest0-dev libxcb-xinerama0-dev libxcb-shape0-dev libxcb-xkb-dev

11. source /home/{username}/bin/compilervars.sh intel64

12. sudo apt-get install python3-pip

13. pip3 install matplotlib numpy scipy qtpy

14. python3 setup.py install –user

15. Use Example folder files as templates to test the installation


Note* The Pol coupling file does not currently work 

