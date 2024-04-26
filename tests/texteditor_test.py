import pytest
from sd.texteditor import TextEditor

def test_texteditor():
    text = TextEditor("Hello, world!")

    assert text.to_string() == "Hello, world!"
    assert len(text.to_string()) == 13, "The length of the text is incorrect: %d" % len(text.to_string())
    assert len(text.lines()) == 1, "The number of lines is incorrect: %d" % len(text.lines)
    text.add_text("\nHow are you?")

    assert len(text.lines()) == 2, "The number of lines is incorrect: %d" % len(text.lines)
    assert text.caret_pos() == 12, "The caret position is incorrect: %d" % text.caret_pos()

def test_updates():
    text = TextEditor("Hello, world!\nHow are you?\nI am fine.")

    text.move_caret("End")
    assert text.caret_pos() == 10, "The caret position is incorrect: %d" % text.caret_pos()
    assert text.caret_line() == 2, "The caret line is incorrect: %d" % text.caret_line()
    text.move_caret("Home")
    assert text.caret_pos() == 0, "The caret position is incorrect: %d" % text.caret_pos()
    text.move_caret("Down")
    assert text.caret_line() == 1, "The caret line is incorrect: %d" % text.caret_line()
    text.move_caret("Down")
    assert text.caret_line() == 2, "The caret line is incorrect: %d" % text.caret_line()
    text.move_caret("Down")
    assert text.caret_line() == 2, "The caret line is incorrect: %d" % text.caret_line()

@pytest.mark.parametrize("direction, expected_pos, expected_line", [
    ("End", 10, 2),
    ("Home", 0, 0),
    ("Up", 0, 0),
    ("Right", 1, 0),
    ("Up", 0, 0),
    ("Down", 0, 1),
])
def test_move_caret(direction, expected_pos, expected_line):
    text = TextEditor("Hello, world!\nHow are you?\nI am fine.")
    text.caret_pos(0)
    text.move_caret(direction)
    assert text.caret_pos() == expected_pos, "The caret position for %s is incorrect: %d" % (direction, text.caret_pos())
    assert text.caret_line() == expected_line, "The caret line is incorrect: %d" % text.caret_line()
