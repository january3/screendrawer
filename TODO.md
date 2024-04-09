To do (sorted by priority):


 * unit tests. more, more, more
 * write sdrw2yaml.py to be able to quickly inspect the contents of the sdrw
   files
 * create a pen wiglet
 * make wiglets movable
 * draw a dustbin wiglet in lower left corner
 * clean up font code. Maybe use the Pango.FontDescription class for
   everything - why not?
 * idea for path editing: "thumb" - moving a point on path and dragging
   surrounding points with it in a rubber-like fashion; how many - that depends on current line
   width (so broader line make more points move)
 * zoom in and out. I think the time is ripe for it 
 * brushes. (1) brush that generates short diagonal thin strokes (2) brush
   that creates a Gary Larson-like hatching (3) brush that creates a
 * keys 1-0 should select one of 10 pens; ctrl-1 to 0 should set the pen
   to the corresponding pen
 * wiglets for pen / line width / tool. They should be drawinggroup
   objects knowing how to draw themselves and how to react to mouse
 * Help should be actually a new screendrawer window with written on it!
 * how should the color picker and the color selection dialog colaborate?
 * think hard how I want the color setting / pen thing to work
 * add a line mode and Line object class
 * show corners of the bounding box
 * grid
 * horizontal and vertical guides
 * PDFs should be multipage (ha, ha)

Design issues:
 * the interaction between canvas, gom, dm, em is tangled. 
 * It is not entirely clear where the file is saved. In theory it is, but
   in practice I find myself wondering.
 * should the EM also take care of pre- and post-dispatch actions? Like
   switching to a ceratain mode after or before certain commands

Bugs:
 * grid must be cached
 * export / conversion with an empty page fails
 * when pasting the object, the new object should be placed next to the
   cursor.
 * when paste an object multiple times, the second and following copies
   fall on the same position as the first one
 * when text is grouped with other objects, and the group is resized in
   unproportional way, then due to the fact that the bb of the text is
   resized (more or less) proportionally, after a few resize operations the
   text size is very small.
 * when drawing very slow the line looks like shit.
 * rotating the whole selection does not work (b/c the way selection
   behaves)
 * when text is rotated, the algorithm for checking for hover objects does
   not consider the enlarged bounding box
 * when the bb is smaller than the corner clicking area, bad things happen
   (it is hard to move the object for example) -> the corner clicking area
   should be mostly outside of the bb

Done:
 * regression: bounding box of brush 4 is not calculated correctly
 * regression: ctrl-click for changing line width behaves erratically
 * the caching mechanism in Drawer is not perfect; when objects are cached,
   the cache is always on the bottom of the stack. So for example when
   moving an object underneath, during the move the object is on the top of
   other objects. Basically, the cache should split the "same" objects into
   groups depending on their z-position relative to the active (changing)
   objects and create a cache for each of these groups separately.
 * FIXED: the solution is a Drawer class which keeps a cache surface with
   the objects that are not changing painted upon.

   the cr.stroke() and cr.fill() are really cpu intensive which is a
   problem with drawings containing many strokes. One way of dealing with
   that would be the following: paths can consist of several subpaths. So
   after release-button, if nothing else changed (mode, pen etc.), then the
   next click actually extends the current path. This would result in the
   whole complex drawing to be drawn with one stroke. Downside: ctrl-z
   would undo the whole drawing; also removing individual subpaths would
   not be possible.

   Alternatives: 
    * create a special object, "PathGroup" which would actually do the
      single stroke or fill command after the paths have drawn themselves.
      Not sure how that would work.
    * create a common bus or something scheduling drawing operations. The
      idea would be that the central drawing functions would collect the
      draw events, check whether anything changes between them (like color,
      line width etc.), and then only stroke if necessary.
    * or, maybe, if the central drawing function sees a bunch of paths, it
      analyses them (checking whether they have the same brush type, color
      etc.) and if yes, it draws them in one go. So basically it asks all
      of them to draw without stroking, and then ask the last of them to do
      the final stroke. This grouping could even be cached and recalculated
      only when something substantial changes.
   All that will not work with brushes that change the color / transparency
   while drawing (e.g. pencil).
 * Another idea: caching objects. Basically, most of the objects do not
   change most of the time. It would be sufficient to create a pixbuf of
   the objects and then draw the pixbuf until the objects change.
   Not sure how to implement it, but it looks like a shitload of work. One
   would probably need to start with recording which objects change from
   one draw operation to another; and if after, say, three draw operations
   the objects do not mutate, we cache them until they change. How to
   detect the change? Maybe by hashing the object properties somehow? so
   each object calculates its own hash. One very simple possibility of a
   hash would be to add a 1 to the hash every time the object changes. This
   would be the objects responsibility.
   The problem with this approach is stacking. Basically, any object
   *after* an object that changed should be redrawn.

   Or, maybe, each object should cache its own pixbuf. Or at least the
   Paths. This might be quicker than redrawing the object every time.
   The advantage would be simplicity of implementation.  => tried that. It
   is not effective enough with many objects. That would make sense for
   few, very complicated paths.
 * I fixed the issue with rotating the brush by scaling / rotating / moving
   the outline instead of recalculating. The side effect is that the
   outline is, well, scaled, which is not what I would like to have: I
   would prefer to have the "line width" (brush outline) not change with
   scaling. Basically the only solution I see is to have the 
 * when brush 3 is rotated, the outline is not recalculated. However, upon
   save + exit the outline is recalculated which results in a modified
   outline. Probably the pen should save the rotation. Or maybe the outline
   should be saved in the object itself.
 * undo remove object places the object in the wrong position in stack - at
   the end of the stack, instead of the exact position that it was located.
   Thus, after the undo operation, the stack is not the same as before the
   operation. This may be a big problem with subsequent undos that actually
   consider the stack order.
 * add -p PAGE parameter to the command line interface
 * rotating / moving of paths recalculates the outline. This is not OK for
   brushes like no. 3, because the outline has been calculated with certain
   absolute slant in mind.
 * cleaned up install dir, created toml file for pip install, pip3 install -e . seems now to work
 * switching between layers should autoselect all objects for a visual hint
 * regression: ctrl-a (select all) no longer works 
 * undo for deleting layers
 * remember canvas translation; separately for each page!
 * clearing up the canvas should clear the whole page, not only the current
   layer
 * unit tests. this is becoming pressing but will be a lot of work
 * extra layers created when importing
 * only current layer gets saved
 * add layers so that we can create a sketch in one layer, then draw in another, and
   finally remove the first one. Layers could be implemented as
   DrawableGroup or Page object within the Page class. Page should then keep track
   of existing layers. The layers list would be independent of any
   selection information, but the selection object and tool would probably
   have to be aware of the layers; ctrl-a should only select objects from
   the current layer.
 * remember page number
 * implemented three simple brushes, selectable through 1-3
 * implemented grid (ctrl-g)
 * implement rotating for: Box, Circle (yes, since Circle can be an
   ellipse)
 * implement undo for fonts as well
 * Replaced "Boxes" with "Rectangles" which are really Shapes (closed
   paths). So I don't need to separately implement rotations for
   Rectangles, which proved to be tricky, esp. for calculating the bounding
   boxes of a rotated box. The Box class is still there, used for example
   for the selection tool.
 * double-shift clicking in text mode results in exceptions being thrown
   around. The problem is that a double click event gets raised before the
   second click is released; therefore, a text object is created with the
   first click, then again with the second click, and then the second click
   raises a double click event, which tries to remove the previous text
   object, but it is not entered into the object list yet.
 * Clicking a text in text mode results in exceptions being thrown around.
 * regression: after short click, the current object does not disappear
   so mouse release is not properly detected
 * circle and box do not respect transparency
 * dragging an object to lower left corner results in an exception
 * something is horribly wrong with undo when group / ungroup is involved.
   to reproduce: create three objects. group them. ungroup them. undo. move
   them around. undo. [found:] the problem is that move command simply
   takes one object and moves / unmoves it. However, when the object is a
   selection, which varying content, the undo will be applied to whatever
   is (or not) selected at the time of the undo. There are two solutions:
   (i) pass lists of "real" objects, so the caller has to take care of
   making sure these are real objects, or (ii) implement "get_primitive" to
   search for real graphic primitives in the object tree and apply the move
   to each of them. [fixed:] The caller uses the copy() method of the
   selection object to get a copy in a DrawableGroup object which is safe
   to keep in the history.
 * deleting page cannot be undone: refactor history so it is run on level
   of GOM and not the individual pages, add a DeletePageCommand. Basically,
   each Command should take the Page object as an argument to know on which
   page to act.
 * moved command history from Page to GOM
 * regression: z-moving no longer works
 * when copying when nothing is selected, simply copy all objects
 * deleting pages
 * converting drawings to png contains a typo; also converting with borders does not
   work correctly
 * add "pages" or "frames" or "slides" or "whatever" to the
   drawing. This would allow to switch between different drawings.
 * implemented a "page" class that serves as an interface between GOM and
   objects. The goal is to create new pages on the fly.
 * add a better color picker, the gtk thing sucks. Something like in
   inkscape could be nice.
 * when creating boxes, sometimes tiny itsy bitsy boxes are created
 * bounding boxes of objects that were reversed during resize are incorrect
 * When exporting, no background is produced
 * wiglet for color 
 * an idea: wiglets which are shown (optionally, toggleable) on the left
   side of the screen, allowing to quickly select colors, line widths,
   transparency etc.
 * paning the draw area
 * maybe an infinite drawing area? Scrollable like?
 * Drawing Manager
 * when converting via command line, it should be possible to specify a
   border around the actual drawing (so not all screen is exported)
 * implement command line conversion between sdrw and (png, svg, pdf, ...)
   -> this will require detaching the drawing from the window!
 * save selection as...
 * "apply pen" -> when run, apply the pen to selection (color, width, etc.)
 * implement "copy pen" or "set with pen" thingy. Also, 
 * changing bg color from current pen?
 * implement a way to change the color of the background
 * export image should use the background color
 * bug in exporting image - regression
 * "save as" dialog
 * selection all should switch to move mode
 * regression: text can't be entered
 * TransmuteCommand handles exchanging objects in the selection
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
   or what? => EM is the boss and knows it all. It has access to GOM and
   the app and knows which methods to associate with which events.
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

Parked ideas:
 * import SVG: that would be a nice thing, but it is a lot of work. Also,
   to do it properly it would require specialized libraries.
 * For outlines, split each outline into non-overlapping segments. This is
   much harder than I thought it would be, but fun.

Rejected ideas:
 * how about: each object has methods "save_state" (which returns
   everything that is needed to completely restore state) and "restore_state"
   (which restores the state). This would allow to save the state of the
   object before the operation and restore it after undo. That way, 
   every object would have to take care of proper restore of its state, and
   MoveCommand, SetLWCommand etc. would only have to call the save_state
   when modifying the objects. Drawback: this would require loads of
   memory, because the object does not know what part of the state is
   needed and which is not. [I think the current solution is OK, that is:
   commands can filter out the primitives with get_primitive and so save the state of their
   properties without an issue]
 * Maybe the *objects* should hold their undo history? This would make
   sense, because the objects are the ones that are changed. However, this
   would require that the objects are aware of the commands, which is not
   the case now.
 * turn it into a Gnome plugin (who 一体 needs that?)
 * eraser should allow a selection like with selection tool (really? what
   for? why not select in select mode and press del?)
 * add laserpointer mode? (why?)
 * loading from SVG (come on. this is already a waste of time)
 * shortcut or menu item for decorating / unmaximizing the window (so it
   can be moved to another monitor) (this is needed in teaching!) -> pity,
   I implemented it, but it doesn't work correctly; there seems to be a bug
   either with Gnome or Gtk. However, if you move the cursor to the other
   monitor and then start sd it opens there, so it is not a big problem.
 * maybe I am doing it all wrong. Maybe I should define a transformation
   class and then record transformations for each object. This way, I
   would be able to undo transformations easily. This is a big design
   issue.




