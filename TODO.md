To do (sorted by priority):

 * implement undo for fonts as well
 * shortcut or menu item for decorating / unmaximizing the window (so it
   can be moved to another monitor)
 * close path: converts a path to polygon (what with the outline / pressure? do we loose it?)
 * "apply pen" -> when run, apply the pen to selection (color, width, etc.)
 * add "pages" or "layers" or "frames" or "slides" or "whatever" to the
   drawing. This would allow to switch between different drawings.
 * implement rotating for: Box, Circle
 * add a better color picker
 * add a line mode and Line object class
 * show corners of the bounding box
 * an idea: wiglets which are shown (optionally, toggleable) on the left
   side of the screen, allowing to quickly select colors, line widths,
   transparency etc.
 * color picker.
 * grid
 * horizontal and vertical guides
 * turn it into a Gnome plugin
 * eraser should allow a selection like with selection tool (really?)
 * add laserpointer mode?
 * loading from SVG

Design issues:
 * maybe I am doing it all wrong. Maybe I should define a transformation
   class and then record transformations for each object. This way, I
   would be able to undo transformations easily. This is a big design
   issue.
 * Maybe we should not be appending the object to the self.objects and the
   command to the history *until* it is not finished? And draw the current
   object separately after all the other objects?

Bugs:
 * proportional resize isn't
 * when the bb is smaller than the corner clicking area, bad things happen
   (it is hard to move the object for example) -> the corner clicking area
   should be mostly outside of the bb
 * when exiting while selection is being made with a box, the selection box
   becomes an object upon new start
 * double click enters text editing only in draw mode, not in text mode
   - the problem is that before a double click event is raised, single
     click events happen which confuse the app.

Done:
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
