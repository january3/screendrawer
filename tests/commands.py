import pytest
from pytest import approx
from unittest import mock
from unittest.mock import patch, MagicMock

from sd.commands import *
from sd.drawable_group import DrawableGroup
from sd.drawable_primitives import Shape, Image, Text, Rectangle, Circle
from sd.drawable_paths import Path
from sd.pen import Pen
from sd.page import Page

def test_color_set_func():
    """Test the color_set_func function"""

    obj = MagicMock()
    obj.pen = MagicMock()
    obj.pen.color = (0, 1, 0)

    # make sure that the color_set method of obj is called
    color_set_func(obj, (0, 1, 0))
    obj.color_set.assert_called_once_with((0, 1, 0))

    assert color_get_func(obj) == (0, 1, 0), "Color not set correctly"

def test_SetColorCommand():
    """Test the ColorCommand class"""

    obj = MagicMock(spec = Shape)
    obj.pen = MagicMock(spec = Pen)
    obj.pen.color = (0, 1, 0)

    obj_group = MagicMock(spec = DrawableGroup)
    obj_group.get_primitive.return_value = [ obj ]

    cmd = SetColorCommand(obj_group, (1, 0, 0))
    obj.color_set.assert_called_once_with((1, 0, 0))

# ----------------------------------------------------------------------------
def _mk_obj(sel = None):
    testobjects = {
            "image": Image(coords = [(0, 0), (100, 100)], pen = Pen(color = (0, 1, 0)), image = None,
          image_base64 =
"iVBORw0KGgoAAAANSUhEUgAAABwAAAAbCAYAAABvCO8sAAACm0lEQVRIx7XWT4hVVRgA8N+ZGRuFnkgoEhMTITbZJsEg7I8WVIsQ0UELFxGNk6KbmFYtEqIsApNW4mbmRRRJi+i1UrNCFyahhbkIF9ngFKMp40yMohX2tTkjt/G+8s17c+Bu7v3O/XG+e75zv2SWRkTE9HsppZUds4WllL7GZQxgCVZHxLGOWcKOYSyqlY1YC6lv8hC0NfPi4ihgx3EqY8UxHzqaWMUr2IyfUMv3DuBoVCtvlkxbDKkJ8Aj2YAPOYByrolpZc9Nm6Zv8AQsjoqutic/1I3bhT9yN3jrYW5gbEV0ppZSa3Bz7sDTvhe/xYFQrywvY53gkIhY2Y03V2Bpcwa/4Aidziv+KaiUwgi/L6nEmWC+uZ/BevJjRU9iBg9jfKmwFruEqni48fgz7MJQ30gc4NGM0T1yGCzmND5SEPZdX+Q6ewi8zAvOkroz9hnvqhPbgW5zHaYw0DOYJnfgZO9FdJ3QAYzm+He/i44bAHNyNsxmrN7oLmOlHXiPYC/nE3wn9ajF1FULvK6Su/f/em/6jqPtzWjbgq2kIGLTuCXyUT5HOfOo0BmbsNTyLT/B2GTa0YPtnMTF6f0T0pJTa8fetZK6tBHs519RDnds+/LQMe7/79RMxMbooY3NvFfsXmLEBrMfG+YNx1/N7K6dvwnreO3t95OSliHg0pXQb/mhk16cC9hL68GS/2pWy4KFFAxfi4vDRiFjfSBrrpXQrnqmHVe989VpcHK5lLM0Eg468ut040a82Xrqy2zdPxrmxXRHxRlO/mUKLsVZq61VSpoNzNv3u8tXDGZvXikZr+MYHXfr4xFRhz1m95TspfYPRXNSpFVjCJdxRaBvm4RwWRMSy3MC2BLtRDu1LHh6f+pVMOwc7W923/gMDooiVvplNJAAAAABJRU5ErkJggg==",),
            "text": Text(coords = [(0, 0), (100, 100)], pen = Pen(color = (0, 1, 0)), content = "Test text"),
            "rectangle": Rectangle(coords = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)], pen = Pen(color = (0, 1, 0)), filled = False),
            "circle": Circle(coords = [(0, 0), (100, 100)], pen = Pen(color = (0, 1, 0)), filled = False),
            "path": Path(coords = [(0, 0), (10, 30), (50, 50), (100, 100)], pen = Pen(color = (0, 1, 0), line_width = 0.1), pressure = [1, 1, 1, 1]),
            "shape": Shape(coords = [(0, 0), (20, 30), (50, 50), (100, 100)], pen = Pen(color = (0, 1, 0))),
    }
    if sel:
        return [ testobjects[x] for x in sel ]
    return [ x for x in testobjects.values() ]

@pytest.mark.parametrize("obj", _mk_obj())
def test_SetColorCommand_on_objects(obj):
    """Test the ColorCommand class on actual drawable objects"""

    obj_group = DrawableGroup([ obj ])

    cmd = SetColorCommand(obj_group, (1, 0, 0))
    assert cmd is not None, "Creating SetFontCommand failed"
    assert obj.pen.color == (1, 0, 0), "Color not set correctly"
    cmd.undo()
    assert obj.pen.color == (0, 1, 0), "Undo failed"
    cmd.redo()
    assert obj.pen.color == (1, 0, 0), "Redo failed"

@pytest.mark.parametrize("obj", _mk_obj())
def test_SetPenCommand_on_objects(obj):
    """Test the SetPenCommand class on actual drawable objects"""

    obj_group = DrawableGroup([ obj ])
    pen = Pen(color = (1, 0, 0), line_width = 999)

    assert obj.pen.color == (0, 1, 0), "Undo failed"
    cmd = SetPenCommand(obj_group, pen)
    assert cmd is not None, "Creating SetFontCommand failed"
    assert obj.pen.color == (1, 0, 0), "Pen not set correctly"
    assert obj.pen.line_width == 999, "Pen not set correctly"
    cmd.undo()
    assert obj.pen.line_width != 999, "Undo failed"
    assert obj.pen.color == (0, 1, 0), "Undo failed"
    cmd.redo()
    assert obj.pen.color == (1, 0, 0), "Redo failed"


@pytest.mark.parametrize("obj", _mk_obj([ "text" ]))
def test_SetFontCommand_on_objects(obj):
    """Test the SetFontCommand class on actual drawable objects"""

    obj_group = DrawableGroup([ obj ])

    font = "FooBar bold italic 42"
    cmd = SetFontCommand(obj_group, font)

    assert cmd is not None, "Creating SetFontCommand failed"
    assert obj.pen.font_size == 42, "Font not set correctly"
    assert obj.pen.font_family == "FooBar", "Font not set correctly"
    assert obj.pen.font_weight == "bold", "Font not set correctly"
    assert obj.pen.font_style == "italic", "Font not set correctly"

    font = "BazFoo normal normal 24"
    cmd = SetFontCommand(obj_group, font)

    assert cmd is not None, "Creating SetFontCommand failed"
    assert obj.pen.font_size == 24, "Font not set correctly"
    assert obj.pen.font_family == "BazFoo", "Font not set correctly"
    assert obj.pen.font_weight == "normal", "Font not set correctly"
    assert obj.pen.font_style == "normal", "Font not set correctly"

    cmd.undo()
    assert obj.pen.font_size == 42, "Font not set correctly"
    assert obj.pen.font_family == "FooBar", "Font not set correctly"
    assert obj.pen.font_weight == "bold", "Font not set correctly"
    assert obj.pen.font_style == "italic", "Font not set correctly"

    cmd.redo()
    assert obj.pen.font_size == 24, "Font not set correctly"
    assert obj.pen.font_family == "BazFoo", "Font not set correctly"
    assert obj.pen.font_weight == "normal", "Font not set correctly"
    assert obj.pen.font_style == "normal", "Font not set correctly"


@pytest.mark.parametrize("obj", _mk_obj([ "rectangle", "circle", "shape", "path" ]))
def test_ResizeCommand_on_objects(obj):
    """Test the ResizeCommand class on actual drawable objects"""

    coords = obj.coords[:]
    bb1     = obj.bbox()
    assert bb1 == approx((0, 0, 100, 100), abs = 10), "Bounding incorrect"

    cmd = ResizeCommand(obj, (100, 100), "lower_right", page = 1)
    assert cmd is not None, "Creating ResizeCommand failed"
    cmd.event_update(200, 200)
    cmd.event_finish()

    bb2 = obj.bbox()
    assert bb2 == approx((0, 0, 200, 200), abs = 10), "Bounding incorrect"
    page = cmd.undo()
    assert obj.bbox() == approx(bb1, abs = 1), "Undo failed"
    assert page == 1
    cmd.redo()
    assert obj.bbox() == approx(bb2, abs = 1), "Undo failed"

    cmd = ResizeCommand(obj, (200, 0), "upper_right", page = 1)
    cmd.event_update(100, 50)
    cmd.event_finish()

    bb3 = obj.bbox()
    assert bb3 == approx((0, 50, 100, 150), abs = 1), "Bounding incorrect"


@pytest.mark.parametrize("obj", _mk_obj())
def test_MoveCommand_on_objects(obj):
    """Test the MoveCommand class on actual drawable objects"""

    coords = obj.coords[:]
    bb1     = obj.bbox()
    #assert bb1 == approx((0, 0, 100, 100), abs = 10), "Bounding incorrect"

    cmd = MoveCommand(obj, (100, 100), page = 1)
    assert cmd is not None, "Creating ResizeCommand failed"
    cmd.event_update(200, 200)
    cmd.event_finish()

    bb2 = obj.bbox()
    assert bb2[0] - bb1[0] == 100, "Bounding incorrect"
    assert bb2[1] - bb1[1] == 100, "Bounding incorrect"

    page = cmd.undo()
    assert obj.bbox() == approx(bb1, abs = 1), "Undo failed"
    assert page == 1

    page = cmd.redo()
    assert bb2[0] - bb1[0] == 100, "Bounding incorrect"
    assert bb2[1] - bb1[1] == 100, "Bounding incorrect"
    assert page == 1

def test_RemoveAddCommand():
    """Test the RemoveCommand and AddCommand classes"""

    obj = [ 1, 2 ]
    stack = [ 1, 2, 3, 4, 5 ]

    cmd = RemoveCommand(obj, stack, page = 2)
    assert cmd is not None, "Creating RemoveCommand failed"

    assert stack == [ 3, 4, 5 ], "Remove failed"
    assert cmd.undo() == 2, "Undo failed"
    assert stack == [ 3, 4, 5, 1, 2 ], "Undo failed"
    assert cmd.redo() == 2, "Redo failed"
    assert stack == [ 3, 4, 5 ], "Redo failed"

    cmd = AddCommand(1, stack, page = 3)
    assert cmd is not None, "Creating AddCommand failed"
    assert stack == [ 3, 4, 5, 1 ], "AddCommand failed"
    assert cmd.undo() == 3, "Undo failed"
    assert stack == [ 3, 4, 5 ], "Undo failed"
    assert cmd.redo() == 3, "Redo failed"
    assert stack == [ 3, 4, 5, 1 ], "Redo failed"


def test_ZStackCommand():
    """Test the ZStackCommand class"""

    obj = [ "2", "4" ]
    stack = [ "1", "2", "3", "4", "5" ]

    cmd = ZStackCommand(obj, stack, "raise", page = 2)
    assert cmd is not None, "Creating ZStackCommand failed"

    assert stack == [ "1", "3", "5", "2", "4" ], "ZStack failed"
    assert cmd.undo() == 2, "Undo failed"
    assert stack == [ "1", "2", "3", "4", "5" ], "Undo failed"
    assert cmd.redo() == 2, "Redo failed"
    assert stack == [ "1", "3", "5", "2", "4" ], "Redo failed"

    obj = [ "2", "4" ]
    stack = [ "0", "1", "2", "3", "4", "5" ]

    cmd = ZStackCommand(obj, stack, "lower")
    assert stack == [ "0", "2", "4", "1", "3", "5" ], "ZStack failed"

    stack = [ "0", "1", "2", "3", "4", "5" ]
    cmd = ZStackCommand([ "3" ], stack, "top")
    assert stack == [ "0", "1", "2", "4", "5", "3", ]
    cmd = ZStackCommand([ "3" ], stack, "bottom")
    assert stack == [ "3", "0", "1", "2", "4", "5" ]

    # the following should fail
    with pytest.raises(Exception):
        ZStackCommand([ "3" ], stack, "foo")

    with pytest.raises(Exception):
        ZStackCommand([ "42" ], stack, "top")

def test_GroupObjectCommand():
    """Test grouping objects"""

    stack_orig = _mk_obj()

    stack_cpy = [ x for x in stack_orig ]
    assert len(stack_cpy) == 6, "Copying objects failed"

    obj = [ stack_orig[0], stack_orig[1] ]

    cmd = GroupObjectCommand(obj, stack_cpy, page = 1)
    assert cmd is not None, "Creating GroupObjectCommand failed"
    assert cmd.command_type() == "group", "Command type incorrect"

    assert len(stack_cpy) == 5, "Grouping failed"
    assert isinstance(stack_cpy[0], DrawableGroup), "Grouping failed"
    stack_cpy2 = [ x for x in stack_cpy ]

    grouped_obj = stack_cpy[0].get_primitive()
    assert len(grouped_obj) == 2, "Group too large"
    assert grouped_obj[0] in obj, "Grouping failed"
    assert grouped_obj[1] in obj, "Grouping failed"

    assert cmd.undo() == 1, "Undo failed"
    assert stack_cpy == stack_orig, "Undo failed"
    assert cmd.redo() == 1, "Redo failed"
    assert stack_cpy == stack_cpy2, "Redo failed"

def test_GroupObjectCommand():
    """Test grouping objects"""

    stack_orig = _mk_obj()
    grouped = DrawableGroup([ stack_orig[0], stack_orig[1] ])
    stack_grouped = [ grouped, stack_orig[2], stack_orig[3], stack_orig[4], stack_orig[5]]
    stack = stack_grouped[:]
    assert len(stack) == 5, "Copying objects failed"

    cmd = UngroupObjectCommand([ grouped ], stack)
    assert cmd is not None, "Creating UngroupObjectCommand failed"
    assert cmd.command_type() == "ungroup", "Command type incorrect"
    assert len(stack) == 6, "Ungrouping failed"
    assert stack_orig[0] in stack, "Ungrouping failed"
    assert stack_orig[1] in stack, "Ungrouping failed"

    cmd.undo()
    assert stack == stack_grouped, "Undo failed"
    cmd.redo()
    assert len(stack) == 6, "Ungrouping failed"
    assert stack_orig[0] in stack, "Ungrouping failed"
    assert stack_orig[1] in stack, "Ungrouping failed"

def test_DeletePageCommand():
    """Test deleting pages"""

    page1 = Page()
    page2 = page1.next()
    page3 = page2.next()

    assert page1.next() == page2, "Creating pages failed"
    assert page2.prev() == page1, "Creating pages failed"
    assert page2.next() == page3, "Creating pages failed"
    assert page3.prev() == page2, "Creating pages failed"

    cmd = DeletePageCommand(page2)
    assert cmd is not None, "Creating DeletePageCommand failed"
    assert cmd.command_type() == "delete_page", "Command type incorrect"

    assert page1.next() == page3, "Deleting page failed"
    assert page3.prev() == page1, "Deleting page failed"

    cmd.undo()
    assert page1.next() == page2, "Undo failed"
    assert page2.prev() == page1, "Undo failed"
    assert page2.next() == page3, "Undo failed"
    assert page3.prev() == page2, "Undo failed"

    cmd.redo()
    assert page1.next() == page3, "Redo failed"
    assert page3.prev() == page1, "Redo failed"
    cmd.undo()

    cmd = DeletePageCommand(page1)

    assert page2.prev() == page2, "Deleting page failed"
    assert page2.next() == page3, "Deleting page failed"
    assert page3.prev() == page2, "Deleting page failed"

    cmd.undo()
    assert page1.next() == page2, "Undo failed"
    assert page2.prev() == page1, "Undo failed"
    assert page2.next() == page3, "Undo failed"
    assert page3.prev() == page2, "Undo failed"

    cmd = DeletePageCommand(page3)
    assert page1.next() == page2, "Deleting page failed"
    assert page2.prev() == page1, "Deleting page failed"
    assert page2.next(create = False) == None, "Deleting page failed"


def test_DeleteLayerCommand():
    """Test deleting layers"""

    page = Page()
    assert page.number_of_layers() == 1, "Creating layers failed"
    assert page.next_layer() == 1, "Creating layers failed"
    assert page.next_layer() == 2, "Creating layers failed"
    assert page.prev_layer() == 1, "Creating layers failed"
    assert page.number_of_layers() == 3, "Creating layers failed"

    cmd = DeleteLayerCommand(page, 1)
    assert cmd is not None, "Creating DeleteLayerCommand failed"
    assert cmd.command_type() == "delete_layer", "Command type incorrect"
    assert page.number_of_layers() == 2, "Deleting layers failed"

    cmd.undo()
    assert page.number_of_layers() == 3, "Undo failed"

    cmd.redo()
    assert page.number_of_layers() == 2, "Redo failed"
