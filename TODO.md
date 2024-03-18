 * add a better color picker
 * add any kind of font picker
 * add moving and rotating with keyboard
 * implement rotating for: Box, Circle
 * show corners of the bounding box
 * grid
 * horizontal and vertical guides
 * z position
 * loading from SVG
 * turn it into a Gnome plugin
 * eraser should allow a selection like with selection tool (really?)

Design issues:
 * maybe I am doing it all wrong. Maybe I should define a transformation
   class and then record transformations for each object. This way, I
   would be able to undo transformations easily. This is a big design
   issue.

Bugs:
 * when the bb is smaller than the corner clicking area, bad things happen
 * When grouped, the bounding box of the group is incorrect until next
   recalculation
 * paste cannot be undone
 * when exiting while selection is being made with a box, the selection
   box becomes an object upon new start
 * pasting text into a new object ends with cursor set
 * double click enters text editing only in draw mode, not in text mode
   - the problem is that before a double click event is raised, single
     click events happen which confuse the app.

Done:
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
