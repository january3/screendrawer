# screendrawer

Draw on the screen with Gnome and Cairo. Quick and dirty.

I needed a simple tool for drawing on the screen. The existing Gnome plugin
had some compatibility issues and anyway was not ideal for my needs. So I
wrote this simple Python script. It starts up quickly, supports pressure sensitive
drawing and has a few simple commands. Most importantly, since I am the one
who wrote it, it suits my needs perfectly.

# Features

 * Draw on the screen
 * Pressure sensitive drawing
 * Mostly single keystroke commands
 * Save the drawing as PNG
 * Change the font size
 * Change the brush size
 * Write text
 * Insert images
 * Move, group, resize, erase objects
 * Change drawing modes

Here is the 
[Greater Egyptian jerboa](https://en.wikipedia.org/wiki/Greater_Egyptian_jerboa), _Jaculus orientalis_, drawn with
screendrawer (from a photograph, not from life):

![Jaculus orientalis](jaculus_orientalis.svg)

# Installation

## Dependencies

In Ubuntu, install the following packages:

```bash
sudo apt-get install python3-gi python3-gi-cairo gir1.2-gtk-3.0
pip3 install pyautogui
```

*In theory* it should work on other systems as well, but I haven't tested
it. I tried to used platform independent libraries, but I can't guarantee
it will work. I have tested it on a Windows installation, and it seems to
work there as well, but I did not run any thorough tests.

## Install

### Simple install

It's just a single script.

```bash
cp sd.py ~/bin/sd
chmod a+x ~/bin/sd
```

### Install with pip

Run the following command to install the script with pip:

```bash
pip3 install .
```

Now, the reason why I primarily use a single script is that I do not
understand fully the pip build system. It seems to be very complex, and I
am a bit reluctant to rely on it. However, the sd.py script is the result
of processing the `sd/*.py` files with the `scripts/make.py` script, so it should
be equivalent to the pip installation.

### Debian package

Alternatively, get the [latest release](https://github.com/january3/screendrawer/releases/latest) as a
Debian package or a zip file.


# Usage

## Basic usage

Run the `sd` command to start drawing on the screen

Switch between modes using single keys:

 * `d` to draw
 * `m` to move, resize, group / ungroup objects
 * `t` to write text
 * `e` to erase objects
 * `b` (box) to draw a rectangle
 * `c` (circle) to draw a circle

Other commands:

 * F1 / h / ? to show help
 * `Ctrl+Q` or `x` to quit
 * `Ctrl+S` to save the drawing as PNG
 * `Ctrl+I` to open an image
 * `Ctrl+Plus` and `Ctrl+Minus` to change the font size / line width
 * `l` or `Ctrl+L` to clear the screen
 * Ctrl+click to change the brush size (move left / right to change the size)
 * Shift+click to write a text, or change to text mode (`t`) and click to write text
 * grab an object with right mouse button to move it around, or change to
   move mode (`m`) and click to move an object
 * drag the object to the left lower corner to delete it

In `m` mode, you can move objects around, group / ungroup them (`g` and `u`
keys, respectively), resize them, or delete them.

The state is saved in / loaded from `savefile` (on Ubuntu, in the
`~/.local/share/ScreenDrawer` directory) so you can continue drawing later.

## Some differences to other programs

The differences to other drawing programs are intentional and are based on
my own workflow. Here are some of them:

 * The state is saved automatically in the current file. There is no "save"
   command, not really. By default the file is `savefile` in the app
   directory (on Ubuntu, `~/.local/share/ScreenDrawer/savefile`). This
   allows me to quickly start and stop drawing without bothering with
   saving the state. Since the whole program is pretty quick to start and
   load the drawing, it is as good as putting the program in the
   background.
 * "Save as" simply changes the location of the savefile. 
 * If no objects are selected, ctrl-c simply copies all objects. This
   speeds up the workflow. For example, you can produce quickly a
   screenshot (Ctrl-Shift-f), annotate it with the draw tool, hit ctrl-c and paste it
   somewhere without bothering to select anything.
 * The drawings are organized in *Pages*. You can switch between pages with
   `Shift-n` and `Shift-p` keys, and delete them with `Shift-d`. While all
   pages are saved in the savefile, only the current page is displayed, and
   export works only on the current page (this might change in the future).
 * Resizing does not change the width of the paths. This is on purpose (it
   would be trivial to make it proportional). I actually went quite a long
   way to make it work this way. One of the reasons is this: when you
   resize a drawing not proportionally, vertical paths are resized
   differently from horizontal paths. This looks butt ugly.
 * Applying pens, colors, brushes etc. is wildly inconsistent. The reason
   for that is, again, simplification of my own work flow. So for example,
   changing the brush of the current pen does not change the brush of the
   selection, while changing the color does. This is because I never want
   to change the brush of the existing line, but I sometimes wish to change
   its color. (However, you can always apply the current pen to a selection
   with alt-p).
 * The app *never* asks for confirmation when saving files. You can always
   overwrite the file you are saving your screenshot in. It is a feature,
   because I often want to save the screenshot in the same file, and I
   am annoyed by the dialog asking me if I want to overwrite the file.
   Also, I tend to press OK without reading the warning. So beware of that.



## Problems

Try to remove the savefile if you have problems.

### Common issues

 * **The exported PNG has black background.**. The background is not black,
   but transparent. You have exported the graphics while the background was
   transparent. Hit ctrl-b two times to get solid background and then
   export the image again.
 * **The background is ugly beige color.** I am afraid this is a feature,
   because a brownish background is often used to draw with both charcoal
   (black) and chalk (white). You can change the background color with
   ctrl-shift-k or by shift-clicking on the color selector bar.


# Development

## Design principles

 * ONE FILE. No dependencies beyond widely spread modules, no installation, no configuration
 * Transparent, sticky, not decorated window
 * Must start up and exit quickly
 * Everything that could be done with a menu must have a shortcut. The
   opposite does not need to be true.
 * This is not inkscape or illustrator, it does not have loads of features.
 * Single keystrokes and simple shortcuts are preferred

## Code organization

The code is *intended* to be in a single file, `sd.py`. However you can't
conveniently develop a 4K line script in a single file. So I use a hybrid
approach:

 * package `sd` contains most of the code - classes, utils etc.
 * The file `sd_devel.py` is a single file that imports the `sd` package and
   runs the main loop. It is used for development.
 * the Python script `make.py` compiles the package into a single script called
   `sd.py` 


