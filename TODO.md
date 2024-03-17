 * horizontal and vertical guides
 * z position
 * export to SVG, loading from SVG
 * turn it into a Gnome plugin

Bugs:
 * when you make a path small, it loses points. Resizing should not modify
   the original coordinates vector, but rather only the outline.
 * paste cannot be undone
 * when exiting while selection is being made with a box, the selection
   box becomes an object upon new start
 * pasting text into a new object ends with cursor set
 * double click enters text editing only in draw mode, not in text mode
   - the problem is that before a double click event is raised, single
     click events happen which confuse the app.

Done:
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
