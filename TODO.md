To do (sorted by priority):


 * save selection as...
 * Help should be actually a new screendrawer window with written on it!
 * implement "copy pen" or "set with pen" thingy. Also, how should the
   color picker and the color selection dialog colaborate?
 * add a method to change the color of the background
 * think hard how I want the color setting / pen thing to work
 * implement undo for fonts as well
 * implement command line conversion between sdrw and (png, svg, pdf, ...)
   -> this will require detaching the drawing from the window!
 * "apply pen" -> when run, apply the pen to selection (color, width, etc.)
 * add "pages" or "layers" or "frames" or "slides" or "whatever" to the
   drawing. This would allow to switch between different drawings.
 * implement rotating for: Box, Circle (yes, since Circle can be an
   ellipse)
 * add a better color picker, the gtk thing sucks. Something like in
   inkscape could be nice.
 * add a line mode and Line object class
 * show corners of the bounding box
 * an idea: wiglets which are shown (optionally, toggleable) on the left
   side of the screen, allowing to quickly select colors, line widths,
   transparency etc.
 * grid
 * horizontal and vertical guides
 * maybe an infinite drawing area? Scrollable like?

Design issues:
 * the app main window code should be split into UIManager, DrawManager and
   EventManager, which would communicate via callbacks.

   OK, so I implemented the Graphical Object Manager aka GOM. It takes care
   of the object list as well as the selection (because several
   selection-related operations, like "next object" or "select all" require
   access to the object list).

   Next task is the EventManager, which should take care of the mouse
   related events mostly (keyboard events seem to be quite a different
   thing and the central hub I have with the satellite and GOM functions
   seems to work well). However I am still not sure what parts should
   belong to the EventManager and how the communication between the
   managers should look like. Does the EM get access to GOM and calls GOM
   methods? Or does GOM register itself with EM and EM calls GOM methods?
   or what?
 * maybe I am doing it all wrong. Maybe I should define a transformation
   class and then record transformations for each object. This way, I
   would be able to undo transformations easily. This is a big design
   issue.

Bugs:
 * the undo is, I think, still buggy. 
 * undo remove object places the object in the wrong position in stack - at
   the end of the stack, instead of the exact position that it was located.
   Thus, after the undo operation, the stack is not the same as before the
   operation. This may be a big problem with subsequent undos that actually
   consider the stack order.
 * When exporting, no background is produced
 * rotating the whole selection does not work (b/c the way selection
   behaves)
 * when creating boxes, sometimes tiny itsy bitsy boxes are created
 * when text is rotated, the algorithm for checking for hover objects does
   not consider the enlarged bounding box
 * bounding boxes of objects that were reversed during resize are incorrect
 * when the bb is smaller than the corner clicking area, bad things happen
   (it is hard to move the object for example) -> the corner clicking area
   should be mostly outside of the bb

Done:
 * autosave
 * copy of an image does not work
 * move does not undo????
 * close path: converts a path to polygon 
   (what with the outline / pressure? do we loose it? - yeah, we do)
 * When font choice dialog is clicked when a text is being edited, the text
   font is not changed
 * for some reason, boxes around images are filled with black [cannot
   reproduce]
 * command line loading of drawings does not work
 * question: if a file is opened with ctrl-o, should the modifications go
   into that file or in savefile?
 * regression: loading data from the savefile no longer works :-(
 * regression: exception thrown when no savefile present
 * seleting all / reverse etc. does not work (regression after introducing GOM)
 * pasting after cutting internal objects does not work
 * internal pasting cannot be undone
 * select all, reverse selection do not work
 * change the way selectionObject works -> it should be a selectionManager
   rather than a transient object
 * text does not save the rotation
 * paste image can't be undone
 * implement argparse
 * regression: images can't be pasted
 * Screenshot tool should automatically select the closest rectangle
 * double click enters text editing only in draw mode, not in text mode
   - the problem is that before a double click event is raised, single
     click events happen which confuse the app, because it starts typing a
     text.
 * when exiting while selection is being made with a box, the selection box
   becomes an object upon new start
 * proportional resize isn't
 * color picker.
 * make a screenshot tool (create rectangle, snap!)
 * Maybe we should not be appending the object to the self.objects and the
   command to the history *until* it is not finished? And draw the current
   object separately after all the other objects?
 * regression: selection tool no longer works
 * grouping changes the order of the objects, should sort the objects first
   by their poosition in the object list
 * Polygon should be smoothed like the path
 * Cannot undo color changes, and it is a fundamental problem, because I
   change the colors of drawable objects via a command, and commands act
   also on object groups (which belong to the same superclass as drawable
   objects), and groups pass the color change to the individual objects and
   other groups, so command doesn't really keep track of all the changes
   (after all, every object in the group can have different color). So
   either we need to pass the command down the group structure, resulting
   in a CommandGroup that can be undone in one go, or the objects must have
   their own history, which is nonsense. Or is there another solution?
   [ solved by adding a method to the DrawableGroup which returns a
   flattened list of primitives contained in that group, and then storing
   the color change for each of the primitives ]
 * add a Polygon class for drawing closed shapes
 * shift-click on an object does not add to selection
 * right-click context menu when clicked on a part of selection only
   affects the object under the cursor, not the whole selection
 * z position: add moving up and down with alt+up and alt+down (undoable,
   so with the Command class)
 * first pressure point on a line is 0 regardless.
 * pasting text into a new object ends with cursor set
 * paste cannot be undone
 * When grouped, the bounding box of the group is incorrect until next
   recalculation
 * add any kind of font picker
 * add rotating with keyboard
 * add moving with keyboard
 * paths do not report correct bounding box initially.
 * rotating / scaling text or images grouped with other stuff is buggy. The reason is
   that there is no real good function to calculate the scaling when the
   object is rotated. This is not a problem with Paths, because with Paths
   we simply rotate the coordinates.
 * export to SVG
 * inverse selection
 * implement rotating for: Image
 * create a pen class which allows to switch between different pens
 * set transparency with a tool similar to setting line width
 * incorrect bb when text is rotated
 * multiple rotations of text in a group cause it to jump slightly around.
   This is probably due to the fact that the bounding box is not recalculated
   after rotation.
 * after rotating, bounding box is not updated
 * rotating should work like this: the obect can take arbitrary center for
   rotation. This is the center of the bounding box. However, for rotating
   a group this will be a different point.
 * when you make a path small, it loses points. Resizing should not modify
   the original coordinates vector, but rather only the outline.
 * undo
 * something is rotten with saving the bounding box of the images when
    images are scaled.
 * when clicking, more than one path is created, with short "stubs" with
   just a single point.
 * something weird is happening with colors when copying and pasting
   - instead of pasting the object, we paste the image generated from the
     object. *And* something weird is happening with the colors!
 * copy and paste groups and paths
 * grouping

Rejected ideas:
 * turn it into a Gnome plugin (who 一体 needs that?)
 * eraser should allow a selection like with selection tool (really?)
 * add laserpointer mode? (why?)
 * loading from SVG (come on. this is already a waste of time)
 * shortcut or menu item for decorating / unmaximizing the window (so it
   can be moved to another monitor) (this is needed in teaching!) -> pity,
   I implemented it, but it doesn't work correctly; there seems to be a bug
   either with Gnome or Gtk. However, if you move the cursor to the other
   monitor and then start sd it opens there, so it is not a big problem.



