#!/bin/sh

cd $(pwd)/tests && python3 -m unittest discover -v
