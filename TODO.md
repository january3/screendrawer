To do (sorted by priority):

 * brush calculations should be done differently. basically: use vectorized
   numpy operation throughout the code; until object is finished, keep the
   arrays in memory, so that key calculations (e.g. normal vectors) do not have to be repeated for
   everything, but rather just for the last point. Signal the brush when
   a new point is added; signal the object when it is finished; object then
   signals the brush to finish, which then destroys the interim
   calculations. Then the brush saves its outline in the save file.
 * make sure that in the outline mode, everything is cyan
 * drawable should not need pen as mandatory argument; rather, they should
   generate a default pen if none is present
 * For layers to be truly useful, we need more layer properties (alpha,
   visible / non-visible), also rearranging layers. But what kind of
   interface should do that? I would hate a gimp like menu that takes half
   of the screen.
 * remember last file name exported and last directory in which it was
   exported, present this option to the user... but how exactly? sometimes
   we want the current directory.
 * while objects are being moved, there is no need to update the cache -
   just move it around.
 * global preferences file
 * properties dialog called with right-click on an object
 * make a "recent colors" (or "recent pens") widget
 * think hard how I want the color setting / pen thing to work
 * unit tests. more, more, more
 * different brushes should have different cursors (ha, ha, haahaha)
 * clean up font code. Maybe use the Pango.FontDescription class for
   everything - why not?
 * save history! Because why the fuck not (because it is a lot of work)
 * idea for path editing: "thumb" - moving a point on path and dragging
   surrounding points with it in a rubber-like fashion; how many - that depends on current line
   width (so broader line make more points move)
 * keys 1-0 should select one of 10 pens; ctrl-1 to 0 should set the pen
   to the corresponding pen
 * wiglets for pen / line width / tool. They should be drawinggroup
   objects knowing how to draw themselves and how to react to mouse
 * Help should be actually a new screendrawer window with written on it!
 * welcome screen as screendrawer pages
 * show corners of the bounding box
 * horizontal and vertical guides
 * implement page dnd rearrangements in the page selector wiglet
 * make wiglets movable
 * brushes. (1) brush that generates short diagonal thin strokes (2) brush
   that creates a Gary Larson-like hatching (3) brush that creates a
 * use the fitz / pymupdf library to annotate PDFs. This can be done by
   creating images using cairo and then adding them to the PDF; while the
   loaded PDF would constitute the bottom layer of the image, not rendered
   to the image when producing the additions to the PDF.
 * use fitz / pymupdf to import PDFs as images on the current page(s).

Design issues:
 
 * coords should be a class, on which one could do different things. this
   class could use different internal representations, while being able to
   use the same interface. It would take over coordinate transformations,
   bbox calculation and much more, all in one place without the need to do
   it separately in various places. Imagine coords.moveto(...),
   coords.addpoint() or coords.bbox(). That would allow efficient caching
   of the bbox, for example.
 * loading from a dict: rather than passing the keys as arguments to the 
   constructor, pass the dict as a single argument. This will make it easier
   to add new properties to the objects, limit the number of arguments etc
   etc.
 * rewrite everything in rust with gtk-rs
 * when saving as, set also the export directory (unless set previously)
 * use numpy for all coordinates?
 * export bitmap resolution will have to be dealt with accordingly.
   Probably we need to make a custom dialog, with a button to open file
   selection dialog - a bit like it is in inkscape, but not quite. At least
   the "upscaling factor" should be configurable (right now it is rigidly
   defined in export_image_png).
 * clipboard pasting should be taken over by the clipboard class, but for
   that the class would need access first to bus (to emit add object
   events), but also to state, because when pasting text it should know
   whether the current object is text. 
 * when bus emits a signal, it calls the listener without telling it to
   which signal it reacts. This makes it impossible to create
   multi-listeners that handle different signals. Maybe the signal should
   be passed as an argument to the listener? That would require a lot of
   work - every listener must take an additional, optional argument.
 * the logic behind automated groups is as follows: while automated group
   is created, there is a method listening in on any events. And anything
   that is not on the ignore list causes the current group to be finished.
   This should ensure that e.g. we don't change the page while drawing, yet
   certain operations should be still possible. Maybe a more efficient
   alternative would be smth like "notify me if anything *but* these
   specific events happens'. That might save some time on function calls.
 * all shit and their family goes into a mouse event. Isn't that too much?
   Should it not be defined clearer, who needs what from a mouse event?
 * brushes should better cache the calculations. Only real changes should
   trigger the recalculation of brush outlines.
 * maybe numpy should be used for brush calculations.

Bugs:
 * pencil smoothing of segments makes problems with pressure, resulting in
   apparent hard pressure thresholds - but this is just an artifact
 * bbox of the brush is repeatedly finding min and max elements from the
   array. Actually, at least for moving, it is enough to change the first
   two members of `__bbox`
 * ESC should deselect current selection
 * segments are not working smoothly (why? don't remember)
 * why would moving cause to recalculate the outline of the brush???
 * grid is only drawn in the central region.
 * a clipped circle drawn during resize has incorrect coordinates. =>
   problem with "actual" and "not actual" bboxes.
 * when clipping, sometimes the bounding box is adapted to the clipping group for
   some reason. => right, that's because the overlap is not the same as the
   bbox of the clipped region. => look up cairo context, maybe one can get
   the clipping region somehow.
 * remove the frame after taking screenshot
 * select a number of objects, hit alt-s, undo, hit alt-s again -> error.
   something is wrong with selection (old object? containing deleted
   objects?)
 * ctrl-v / ctrl-c do not work while editing text objects
 * Brush two sucks.
 * when paste an object multiple times, the second and following copies
   fall on the same position as the first one. 
 * when drawing very slow the line looks like shit.

Done:
 * in `calc_outline_rounded`, the coordinates of the ends and starts around
   an arc joint are unnecessarily doubled
 * bug that appears when moving the drawing; could not be replicated, here
   is the trace:
   ps: it can be replicated by scribbling in place
 * exported images have tiny resolution. => increased bbox 3 times for
   export as an interim solution
 * exporting bbox: right now, in the uibuilder code, the bbox is specified
   as the screen, unless objects are selected. The rationale is "wysiwyg",
   you can get multiple pages of precisely the same size. However, on the
   other hand, it may be annoying. Then again, hit ctrl-a and you are good.
   => not really, as this only selects from the current layer. => added a
   checkbox to the export dialog
 * only current layer gets exported!!!!
 * brush 4 (pencil) doesn't work anymore, probably due to changing of the
   coord calculation code.
 * pen no 5 (pencil) has incorrect bounding box
 * FUCK. there are bezier curves implemented already in cairo. Was all that
   work with brushes a waste of time? Well, on the other hand, this whole
   project is a waste of time (this hole project is a waste of tame).
 * ctrl-v when cursor hasn't moved places object at (0, 0) instead of the
   actual cursor pos (should be the same as with crosslines) => that's
   because only the absolute (screen) position can be queried at the
   moment, not the user (drawing) position. => why not use
   page.trafo().apply_reverse()?
 * the interaction between canvas, gom, dm, em is tangled.  => can safely
   say that it is tangled in a much different way
 * the cacheing is still suboptimal. In essence, if an object has not
   changed and redrawing is not requested, there is no need to query its
   bbox and check whether it is within the screen. => solved by a better
   bbox reporting mechanism
 * zooming too much results in a cairo error. The reason for that is that
   irrespective of the zoom, we are trying to redraw the whole image. We
   need to clip with the window size instead. Or at least set a cap on
   zoom.
 * when an sdrw file is opened or saved-as, set the working dir to that
   files dir. When a file is exported, remember the dir for exports only. 
   When started in headless mode, start working in the home dir or a dir
   specified by the config file. Ditto default exports file.
 * when zooming in, lines are grainy, same problem as with text below.
   However, we cannot ditch the cacheing. The solution would be to, of
   course, to draw the cache outside of page transformations.
 * maybe gom should be included in State? Like, one of the superclasses so
   that Stat has all gom methods and you can directly interrogate
   state.page() etc.? Or, alternatively, gom should be created by State,
   and exposed through state.gom().
 * when zoomed in, the text is grainy. This is due to the fact that rather
   than being drawn directly, it is drawn on a surface, then the surface is
   zoomed in... you get the pic. Probably, text should be always drawn
   without intermediate caching on surfaces.
 * when zoomed in, text shivers. Apparently it is not cached. =>
   unnecessary mod flags due to bbox recalculation
 * zoom in and out. I think the time is ripe for it 
 * while resizing / rotating, clipped box looks weird
 * rotating clipped objects uses the incorrect rotation origin, but works
   otherwise
 * clipped objects keep their (incorrect) bbox with respect to moving /
   resizing
 * Separation between State and Setter is non-existent, it is unclear which
   does what.
 * if an object has been cut with ctrl-x, then upon ctrl-v it should be
   inserted at precisely the same position. Therefore, we would need to
   mark the origin of the clipboard more than just "internal"
 * Layers have too many public methods. Maybe it was wrong to move
   everything from GOM to Layer? Shit. Maybe Layers should strictly do
   stuff rather than handle signals. Then again, same could be said about
   pages. Gdzie się nie obrócisz, to dupa z tyłu. Or maybe: layers should
   be creating the actual commands, but GOM should be reacting to the bus
   and putting the commands in history. Oh fuuuuuuuuuuuuuuuuuck.
 * next / prev object (tab, shift-tab) does not work, throws an error
 * implement a command to flush a group of objects to l/r/t/b
 * there is a problem with redoing move commands
 * selection-related bus listeners should be handled by the layer, since
   selection is an object of the layer. active page should activate the
   active layer, the layer should set up its own bus listeneres upon
   activation etc. Or maybe selection should do that? I mean, what does
   selection need the layer for? => not sure, when is the selection object
   deleted / replaced? => ok, so selection is transient, therefore it is
   layer that should be handling these signals
 * everything that page can handle, gom shouldn't
 * there are several page-related functions for which gom serves as a
   wrapper, because gom knows the current page. However, instead create a
   "activate"/"deactivate" method pair for page; pages should then turn
   their bus listeners on / off and handle the page related functions
   themselves. This would take a lot of load off gom and create a better
   separation between gom and page. then again, the same thing could be
   said about selection object? since this is a page thing as well.
 * moves cannot be grouped because they work on a copy of the selection
 * when started, the position reported by cursor is 0, 0, because it has
   not received any mouse events yet. An event would have to be simulated
   or triggered. => this seems to be harder to solve then expected, I can't
   even get code to get the position of cursor in a window! => solved in a
   rather inelegant way, duplicating some functionality like update_pos vs
   update_pos_abs in Cursor. 
 * crosslines not visible immediately after toggling
 * make a "moving guides" thingy, like vertical and horizontal moving line
   (crosslines)
 * duplicate: like ctrl-c ctrl-v, except exactly at the same location
 * also, duplicate method: self.duplicate() for drawables that does the
   deepcopy thing. Also, automatically select it.
 * pasted objects automatically selected
 * when pasting the object, the new object should be placed next to the
   cursor. Maybe. I don't really know.
 * Also, often (depending on
   the page translate) the pasted object will outside of the screen.
 * add cmd line option to not be pinned to all windows
 * use logger for printing clipboard debug msgs
 * something is wrong with moving to top / to bottom, doesn't seem to work
 * when using selection tool, the screen coordinates are not mapped
   correctly onto the page coordinates (is transformation page-specific? it
   should be, if not it is a bug). -> this is because selection tool is
   working in screen coordinates, not in draw coordinates => it should be
   drawn from within the screen transformation
 * when exporting selection, "all pages as pdf" option should be inactive.
 * weird left-hand margin when exporting to png. Possibly something to do
   with page translation. No idea. => this is because of clipping; the bbox
   is fit not to the clipping object, but all objects.
 * within one session, remembering the export dir and export file name
 * saving a graphics: by default, the file type should be read from the
   file extension.
 * identical commands (like move) should be merged
 * the grouping of commands works in that for the user, only one ctrl-z
   does the trick. However, rather than the current solution, history
   should attempt to call some "merge" method on commands somehow
   identified as similar. Then this merge function would take a look and
   decide whether commands can be directly merged into one or whether they
   can be merged into a CommandGroup. For example, multiple sets to a
   property can be safely replaced by the very last command used. However,
   increasing the line width / stroke by 1 should be merged cleverly, as
   one command increasing the line multiple times.
 * text editing cannot be undone
 * stroke change does not update the bbox
 * when setting transparency / line width with the UI, this results in
   hundreds if not thousands of undo events. However, I have not the
   slightest HOW I could fuse together the history events. Maybe something
   like a status hashtag? or objects hashtag? like a time stamp that
   denotes that the objects affected did not change? One possibility:

    * when a command is added to the history stack, history looks up the
      previous command. It checks
       - whether the hashtag (e.g. from the affected object IDs) is the
         same
       - whether the command is of the same type
      if yes, it combines them in a single CommandGroup object.
 * There is no way of changing transparency or line width of an existing
   object. the ctrl-click and ctrl-shift-click should check that objects
   are underneath and their settings instead of changing the globals.
 * menus should use the bus as well
 * changing pages or layers while creating automatic groups messes them up.
   same for delete page etc.
 * single path objects still get wrapped up in a group. Maybe it would be
   possible for a single-object-group to act like the actual object? Better
   - this should be done by the grouping wiglet; it should silently
   replace the group with the path when the group is finished. so,
   basically:
    * create a group
    * add the first path to the group
    * remove the group from the page
    * add the path to the page
 * fill toggle is low-level, not undoable and does not work always as
   expected
 * property changes are not noticed when applied to a group, because they
   are applied only to the primitives, and the group mod flag stays what it
   was. => maybe groups should be aware of their members?
 * color change (which affects directly the pen) does not update the object
   mod flag, so object is not redrawn
 * duplicates when creating objects - probably due to some automatic
   grouping shit
 * groupes with a single object should be automatically ungrouped
   => problem with automatic grouping: when we create text objects, they
   are "grouped" in single-object groups. However, that means that they
   cannot be edited simply by double-clicking on them! Maybe automated
   grouping only in drawing mode? -> yeah, that would be swell
 * make a little wiglet that shows the file name
 * It is not entirely clear where the file is saved. In theory it is, but
   in practice I find myself wondering.
 * group drawing mode is cool. Maybe that should be the default? i.e., when
   drawing, all the objects drawn without leaving the mode are grouped. And
   ctrl-shift-g simply closes one group and starts another.
 * undo for rotate doesn't work if rotating was done via keypress
 * changing line width with ctrl-- / ctrl-+ can't be undone
 * History should be a separate single instance class that gets called
   through the bus. However: gom must collaborate with history, because
   when undoing, we should return to the given page. But: maybe the current page
   should be held by the state?

   Or, another take: how about gom doing the history internally, OK. BUT:
   rather than adding the page to the command object (which then returns it
   to gom, so gom knows to which page we should change the view), add it
   through the history, so that history records the view and returns page
   no when gom is asking for undoing a command.
  
    * Actually, history could emit a signal "change to page so and so", and
      gom could react to it. We actually almost have that already.
    * Fine, but when an object is added to history, how does history / the
      object / the caller know what page we have? Only gom knows that.

      Maybe gom should emit a signal "the current page is so and so", and
      then the history can react to it, so that when an command comes up,
      history knows on what page that command is happening.
 * why is history in gom? shouldn't it be in the commands? As in, commands
   should actually add themselves to history? -> but how: undo has to call
   on history object, so we would have to pass the history to each command
   that we create. Unless we make history a singleton, and then commands
   can simply create the history object and get always the same instance.
   Or history is one of the "superobjects" like gom. => actually, it makes
   sense, since gom is doing most of the heavy lifting
 * in the grouping mode, you cannot really undo shit. Also, everything
   disappears if you exit without pressing escape or ctrl-shift-g. Also, if
   you clear a canvas. This is because while the grouping has not been
   finished yet, the objects aren't really in the scope of GOM yet, but
   kept private by the grouping wiglet. At least mode changes should
   trigger finishing the group, because otherwise you switch to mode and
   you can't do shit with the object. So, what are the options:
    * simulate the undo history while doing something else (i.e., simulate
      adding objects, but in reality, add them to the group) -> how should
      the group be handled? if we undo to 0, the group will be still in the
      gom, but empty, which shouldn't be
    * add a special history command for adding and removing objects to an
      existing group. We can add the initial object with the group
      container in one go to GOM, then create the command that adds an
      object to group, then add the command object to history => that
      should work
 * There is a problem with the way that the double clicks are handled. The
   50ms delay is actually quite counterproductive; rather than handling the
   single click after 50ms, the responders to single click event should be
   also catching the double click events and e.g. cancel the drawing of the
   current object.
 * when text is grouped with other objects, and the group is resized in
   unproportional way, then due to the fact that the bb of the text is
   resized (more or less) proportionally, after a few resize operations the
   text size is very small. -> this might be handled by resizing the text
   through tranformation rather than font change. However, it is neither
   easy nor the results are spectacular. I think for now we will park it.
 * events in em should go through the bus
 * make a "grouping" mode. All objects that are created during the grouping
   mode are added automatically to the same group. That would allow
   for example producing consistent handwriting that does not need to be
 * add a line mode and Line object class
 * when exporting with ctrl-e there should be selection option to choose
   the format, including pdf vs multipage pdf (and multipage pdf should be
   default)
 * export to pdf only exports the current page. Both options should be
   possible (multi-page and single page) (note to self: customizing the
   gtk file chooser dialog is a world of pain)
 * sometimes when editing text the release-button event does not seem to be
   properly processed and when exiting with "Esc", the object is being
   moved even though mouse button is not down. (can't reproduce)
 * when laptop set to low power and teams are running, the app does not
   work efficiently. Not sure what can be done about that, as it seems that
   it is more of a polling issue. Maybe create a very simple pen with no
   calculations at all and see how it works? -> then again, it is even
   worse in inkscape. (the problem was handling the double clicks, I think)
 * incidentally, undoing a rotation + scaling on shapes does not work
   properly either, the shape lands in the initial position, but is still
   sheared => why? it looks like the operations *are* being undone, but
   with slight errors. (seems to work now)
 * paths drawn with slanted brush report incorrect bounding box (fragments
   are cut by the cache)
 * empty pages break multi-page pdf conversion from command line
 * export / conversion with an empty page fails
 * Segment creation does not stop when switching to a different mode. Same
   for text object creation.
 * Brush no. 4 is not working correctly with a tablet
 * quick strokes sometimes result in an apparent double click. Rather than 
   doing the double click calculations on our own, we should rather 
   be canceling the single click events after a double click. Send a signal
   "cancel_single_click" to the wiglets?
 * quick double clicking sometimes produces too many events leading to a
   race condition betwen WigletCreateText and WigletCreateObject (the
   former catches the double click, the later catches the extra single
   click)
 * after rotating a "thick" path, the bounding box is incorrect (too
   narrow, so not around the outline, but around the path). Moving it a bit
   fixes it.
 * When drawing, sometimes there is no reaction => actually, the reason was
   that the wiglets were taking over the clicks. The reason for that was
   that the event for some reason gets modified and the x/y position can
   change to 0/0, thus the color selector wiglet takes over the click.
 * something is seriously rotten with resizing rotated images. The bounding
   boxes do not seem to be correct, which results in aberrant behaviour esp
   in clips and groups. => ok, the problem is that I am trying to resize
   the image (scaling to another rectangle), but that is not how
   transformations work. putting together rotation and scaling results in 
   shearing. You cannot just put together all rotations and all scalings.
   => that means the object needs to have its own little history of
   transformations and apply them sequentially when drawing => that means
   all sorts of problems with calculating bboxes and so on. Maybe it is
   possible to get the user coordinates from the cairo context?
 * toggling outline does not refresh drawing cache
 * rotating images results in a flashing black background within the
   bounding box of the image
 * add clipping images
 * screenshots should work differently. If a frame is selected, go with it.
   If not, let the user create a frame and register with the bus for object
   completion event. Remove the box afterwards.
 * implement page insert (shift-I) to create a new page after the current
   one.
 * after changing pen color the cache is not updated (probably because the
   objects do not know the pen has changed) => no, the reason is different:
   if drawable primitives are grouped, the primitives are extracted and
   modified, but the DrawableGroup is not updated so it does not "know"
   it has been modified. The drawer however only checks whether Drawable
   group was modified. Potential solutions:
    * directly check in SetPropCommand whether an object in the argument list 
      is a Drawable, and if yes, tell it to modify itself
    * add a callback for the parent from the primitives. More flexible, but
      might be problematic in case of SelectionObjects.
    * somehow pass the call to modify the property through the
      DrawableGroup, but then this raises a question, how do we track the
      modifications in the command
    * or maybe add a "mod" method to Drawable, which simply tells the
      object it has been modified; then, in cmd, call that method for every
      object in the argument list - just to make sure.
 * or maybe ChatGTP is right and DM shouldn't actually do anything except
   of parsing the events and passing them on. Maybe we could pack all event
   information in the MouseEvent object and the state object, and then pass
   information around "hey, there is this and this happening, who wants to
   take it".
 * sort out the remove / add / group / set commands. some are taking
   objects, some are taking object lists, some are taking drawable groups.
   Inconsistent! => more or less did that. The Move / Rotate / Resize
   commands still take a single object, because otherwise it would require
   really a lot of unnecessary workarounds. Live with that.
 * The redo-command may mess up stuff.
 * ungrouping reverses the z-stack
 * unseen wiglets work? They should not.
 * clean up import-export code
 * incorrect bounding box when exporting with text (see mk.sdrw) => oh no,
   this is actually due to paning? => oh no, we are already dealing with
   it?
 * PDFs should be multipage (ha, ha) -> this is really easy, use
   surface.show_page()! or even cr.show_page()
 * end of a pencil line should be rounded.
 * brush-like brush with a tapered end and round start ("taper")
 * write sdrw2yaml.py to be able to quickly inspect the contents of the sdrw
   files; or, better, create a yaml export option.
 * text bbox is incorrectly reported to the method checking whether text is
   clicked
 * when text is rotated, the algorithm for checking for hover objects does
   not consider the enlarged bounding box
 * a weird bug appeared once when editing text; something was seriously
   wrong with the text object; text was behaving erratic when moved and
   looked like having two copies (maybe somehow entered twice in
   gom/page/layer?). [update]: the bug is fairly reproducible upon
   double-click of a text [update]: this is because it gets entered twice
   in the object list, since it becomes curent_obj again when double
   clicked.
 * color picker should be really an invisible wiglet able to change the colors
 * regression: ctrl-[shift]-click no longer works, because the "active
   wiglet" is in dm, whereas wiglet drawing is in canvas. They should be
   treated like any other wiglets... but they take different precedence.
   First the "top" wiglets get the choice to use up the click. Then objects
   on the canvas. Finally - if the canvas is empty under the click - the
   ctrl-[shift]-click wiglets.
 * since rectangles and circles incorrectly report their bounding box, the
   result is that they get clipped in the cached surface. This looks
   really, really bad.
 * grid must be cached
 * when the bb is smaller than the corner clicking area, bad things happen
   (it is hard to move the object for example) -> the corner clicking area
   should be mostly outside of the bb
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
 * remember the last opened tab.
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
 * grid
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
 * eraser should allow a selection like with selection tool (really? what
   for? why not select in select mode and press del?) => because you don't
   have to use the keyboard?
 * how should the color picker and the color selection dialog colaborate?
   (or actually abandon color selection dialog?) => works for now as is
 * rotating the whole selection does not work (b/c the way selection
   behaves). However, you can group, rotate and ungroup, so I will park
   that for now.
 * import SVG: that would be a nice thing, but it is a lot of work. Also,
   to do it properly it would require specialized libraries.
 * import PDF. Imagine replacing xournal in annotating PDFs!
 * For outlines, split each outline into non-overlapping segments. This is
   much harder than I thought it would be, but fun.

Rejected ideas:
 * should the EM also take care of pre- and post-dispatch actions? Like
   switching to a ceratain mode after or before certain commands
 * draw a dustbin wiglet in lower left corner (I don't really use this
   functionality, del is so much easier)
 * create a pen wiglet (that does what exactly...?)
 * gtk text widget for text editing => well a prototype (without proper
   positioning) is working (called WigletEditText2). However, this will not
   be able to show all the text trasformations (rotation, scaling etc).
 * text object editing: ditch that fucker, just create a gtk widget to do
   the job, why on earth would we want to do it ourselves?? => because then
   complex transformations are not WYSIWYG when editing
 * selectiontool in erase mode should remove the selected objects =>
   actually not, because I would prefer to erase the objects along the
   track.
 * Native SVG format. This would be MUCH slower to start.
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




