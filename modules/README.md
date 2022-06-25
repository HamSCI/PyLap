# PHaRLAP Python

A numpy-compatible Python 3 wrapper for the [PHaRLAP][pharlap] ionospheric raytracer.

## Building

In order to build this library, there are a few required build dependencies:

* [PHaRLAP][pharlap] 4.4.1 or later
* Python 3
* NumPy 1.18 or later
* C compiler (i.e. GCC)
* [Intel Fortran Compiler Redistributable][intel_fortran] 2018 or later.

Ensure that an environment variable named `PHARLAP_HOME` is defined, pointing to
the install location of PHaRLAP.

On Windows:

**NOTE**: Windows is not yet full supported.

```
set PHARLAP_HOME=C:\pharlap_4.4.1
```

On Linux & macOS:

```
export PHARLAP_HOME="/opt/pharlap_4.4.1"
```

To build, execute the following command:

```sh
$ python3 setup.py build
```

## Running Unit Tests

Use the `run_tests.sh` script to execute the entire unit testing suite:

```sh
$ ./run_tests.sh
```

## Installation

To install, first ensure that you are able to successfully build the library.
Then, execute the following command:

```sh
$ python3 setup.py install
```

## License

*TBD*

## Credit

* Joshua Vega, WB2JSV <wb2jsv@arrl.net>

[pharlap]: https://www.dst.defence.gov.au/opportunity/pharlap-provision-high-frequency-raytracing-laboratory-propagation-studies

[intel_fortran]: https://software.intel.com/en-us/articles/intelr-composer-redistributable-libraries-by-version
