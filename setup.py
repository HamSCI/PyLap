
import glob
import os
import platform

#import setuptools
import numpy as np
from distutils.core import setup
from distutils.core import Extension


if platform.system() not in ['Linux']:
    raise OSError('Unrecognized or unsupported operating system.')

#
# Configure PHaRLAP.
#
if 'PHARLAP_HOME' not in os.environ:
    raise OSError('The environment variable "PHARLAP_HOME" must be defined.')

if 'LD_LIBRARY' not in os.environ:
    raise OSError('The environment variable "LD_LIBRARY" must be defined.')

if 'PYTHONPATH' not in os.environ:
    raise OSError('The environment variable "PYTHONPATH" must be defined.')

pharlap_path = os.getenv('PHARLAP_HOME')
intel_path  = os.getenv('LD_LIBRARY')
py_path = os.getenv('PYTHONPATH')

if not os.path.isdir(py_path):
    raise OSError('The environment variable "PYTHONPATH" is invalid.')

if not os.path.isdir(pharlap_path):
    raise OSError('The environment variable "PHARLAP_HOME" is invalid.')

if not os.path.isdir(intel_path):
    raise OSError('The environment variable "LD_LIBRARY" is invalid.')

pharlap_include_path = os.path.join(pharlap_path, 'src', 'C')
pharlap_lib_path = os.path.join(pharlap_path, 'lib', 'linux')
# Define native modules.
#

native_modules = []
COMMON_GLOB = glob.glob('modules/source/common/*.c')

def create_module(name, libraries):
    native_modules.append(Extension(
        'pylap.' + name,
        sources=['modules/source/' + name + '.c'] + COMMON_GLOB,
        include_dirs=[np.get_include(), pharlap_include_path, "include"],
        library_dirs=[pharlap_lib_path, intel_path],
        libraries=libraries))
    pass

create_module('abso_bg', ['propagation', 'maths', 'iri2016', 'ifcore', 'imf', 'irc', 'svml'])
create_module('dop_spread_eq', ['propagation', 'iri2016', 'ifcore', 'imf', 'irc', 'svml'])
create_module('ground_bs_loss', ['propagation', 'ifcore', 'imf'])
create_module('ground_fs_loss', ['propagation', 'ifcore', 'imf'])
create_module('igrf2007', ['maths', 'iri2007', 'ifcore', 'imf', 'irc'])
create_module('igrf2011', ['maths', 'iri2012', 'ifcore', 'imf', 'irc'])
create_module('igrf2016', ['maths', 'iri2016', 'ifcore', 'imf', 'irc', 'svml'])
create_module('iri2007', ['iri2007', 'ifcore', 'imf', 'irc', 'svml'])
create_module('iri2012', ['iri2012', 'ifcore', 'imf', 'irc', 'svml'])
create_module('iri2016', ['iri2016', 'ifcore', 'imf', 'irc', 'svml'])
create_module('irreg_strength', ['propagation', 'iri2016', 'ifcore', 'imf', 'irc', 'svml'])
create_module('nrlmsise00', ['maths', 'iri2016', 'ifcore', 'imf', 'irc', 'svml'])
create_module('raytrace_2d', ['propagation', 'maths', 'ifcore','imf','iomp5'])
create_module('raytrace_2d_sp', ['propagation', 'maths', 'ifcore', 'imf', 'iomp5'])
create_module('raytrace_3d', ['propagation', 'maths', 'ifcore', 'imf', 'iomp5'])
create_module('raytrace_3d_sp', ['propagation', 'maths', 'ifcore', 'imf', 'iomp5'])

#
# Create module.
#
#with open('requirements.txt') as f:
 #   requirements = f.read().splitlines()

setup(
        name='pylap',
        packages=['modules/pylap'],
        #install_requires=requirements,
        version='0.1.0-alpha',
        description='A numpy-compatible Python 3 wrapper for the PHaRLAP ionospheric raytracer',
        ext_modules=native_modules
)
