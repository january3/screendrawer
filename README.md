# screendrawer

Draw on the screen with Gnome and Cairo. Quick and dirty.

# Installation

## Dependencies

install the following packages:

```bash
sudo apt-get install python3-gi python3-gi-cairo gir1.2-gtk-3.0
```

## Install

It's just a single script.

```bash
cp sd.py ~/bin/sd
chmod a+x ~/bin/sd
```

# Usage

 * `sd` to start drawing
 * `Ctrl+Q` or `Ctrl+C` to quit
 * `Ctrl+S` to save the drawing as PNG
 * `Ctrl+Plus` and `Ctrl+Minus` to change the font size
 * 'Ctrl+L' to clear the screen
 * Ctrl+click to change the brush size (move left / right to change the size)
 * Shift+click to write a text
 * grab an object with right mouse button to move it around
 * drag the object to the left lower corner to delete it
 * Changing drawing modes:
   - 'd' to change to draw (default) mode
   - 'm' to change to move mode
   - 't' to change to text mode
   - 'e' to change to erase mode (click to remove object)

The state is saved in / loaded from `~/.screendrawer` so you can continue drawing later.

# Problems

Try to remove ~/.screendrawer if you have problems.
