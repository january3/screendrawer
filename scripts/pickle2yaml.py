#!/usr/bin/env python3

## I use it for debugging. By default, screendrawer serializes data using
## pickle, but sometimes manual inspection is needed. For this I use yaml.

## Usage: pickle2yaml.py <pickle_file> [<yaml_file>]
## Usage: pickle2yaml.py -r <yaml_file> [<pickle_file>]

import argparse
import sys
import pickle
import yaml

## check for option -r

reverse = False
if sys.argv[1] == '-r':
    yaml_file = sys.argv[2]
    reverse = True
else:
    pickle_file = sys.argv[1]

# check for output file
out_file = None
if len(sys.argv) > 2:
    out_file = sys.argv[2]

def yaml2pickle(yaml_file, out_file):
    with open(yaml_file, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    
    if out_file is None:
        yaml.dump(data, sys.stdout)
        return

    with open(out_file, 'wb') as file:
        pickle.dump(data, file)

def pickle2yaml(pickle_file, out_file):
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    # if no out file, write to stdout
    if out_file is None:
        yaml.dump(data, sys.stdout)
        return
    
    with open(out_file, 'w') as file:
        yaml.dump(data, file)


if reverse:
    yaml2pickle(yaml_file, out_file)
else:
    pickle2yaml(pickle_file, out_file)
