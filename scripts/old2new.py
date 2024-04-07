#!/usr/bin/env python3

## I had to write this little ad hoc utility for converting old screedrawer
## files to the new format.  It is probably not useful for anything to anyone
## any more..

import argparse
import sys
import pickle
import yaml

## check for option -r
input_f = sys.argv[1]
output_f = sys.argv[2]

## recursively check a dictionary.
## if "line_width" is a value in the dictionary, 
## replace it with a pen dictionary.
def recurse_over_dict(d):
    # check whether it is a list
    if isinstance(d, list):
        for item in d:
            recurse_over_dict(item)
        return

    # check whether it is a dictionary
    if not isinstance(d, dict):
        return

    if d.get("type") is None:
        return
    
    print("recursing, type", d.get("type"))
    if "outline_l" in d or "outline_r" in d or "line_width" in d or "size" in d or "lwd" in d or "color" in d or "fill_color" in d or "font_size" in d:
        d["pen"] = {
                "line_width": d.get("line_width") or 1,
                "fill_color": None,
                "color": d.get("color") or (0, 0, 0),
                "font_size": 24,
                "font_family": "Sans",
                "font_weight": "normal",
                "font_style": "normal",
                "transparency": 1,
        }
        if "line_width" in d:
            del d["line_width"]
        if "fill_color" in d:
            d["pen"]["fill_color"] = d["fill_color"]
            del d["fill_color"]
        if "size" in d:
            d["pen"]["font_size"] = d["size"]
            del d["size"]
        if "color" in d:
            del d["color"]
        if "color" in d:
            del d["color"]
        if "lwd" in d:
            d["pen"]["line_width"] = d["lwd"]
            del d["lwd"]
        if "outline_l" in d:
            del d["outline_l"]
        if "outline_r" in d:
            del d["outline_r"]

    for k, v in d.items():
        print("key=", k)

        if isinstance(v, dict) or isinstance(v, list):
            recurse_over_dict(v)

with open(input_f, 'rb') as file:
    data = pickle.load(file)
print(data.keys())
recurse_over_dict(data["objects"])
data["config"]["pen2"] = {
        "line_width": 1,
        "fill_color": None,
        "color": (0, 0, 0),
        "font_size": 24,
        "font_family": "Sans",
        "font_weight": "normal",
        "font_style": "normal",
        "transparency": 1,
}


data["config"]["pen"] = {
        "line_width": 1,
        "fill_color": None,
        "color": (0, 0, 0),
        "font_size": 24,
        "font_family": "Sans",
        "font_weight": "normal",
        "font_style": "normal",
        "transparency": 1,
}

with open(output_f, 'wb') as file:
    pickle.dump(data, file)
