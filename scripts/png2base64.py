#!/usr/bin/env python3

import base64
from io import BytesIO
import sys
import argparse
import tempfile
import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, Pango, GLib

parser = argparse.ArgumentParser(
        description="Convert a png file to base64")
parser.add_argument("-i", "--input", help="Input file", required=True)
parser.add_argument("-o", "--output", help="Output file for conversion")
args     = parser.parse_args()


inp_file = args.input

with open(inp_file, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")
print(image_base64)



