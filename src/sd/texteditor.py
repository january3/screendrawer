"""
Class for editing text.
"""

import logging                                                   # <remove>
log = logging.getLogger(__name__)                                # <remove>

class TextEditor:
    """
    Class for editing text.
    """

    def __init__(self, text = ""):
        self.__cont = text.split("\n")
        self.__line = 0
        self.__caret_pos = 0

    def __backspace(self):
        """Remove the last character from the text."""
        cnt = self.__cont
        lno = self.__line
        cpos = self.__caret_pos

        if cpos > 0:
            cnt[lno] = cnt[lno][:cpos - 1] + cnt[lno][cpos:]
            self.__caret_pos -= 1
        elif lno > 0:
            self.__caret_pos = len(cnt[lno - 1])
            cnt[lno - 1] += cnt[lno]
            cnt.pop(lno)
            self.__line -= 1

    def __newline(self):
        """Add a newline to the text."""
        self.__cont.insert(self.__line + 1,
                            self.__cont[self.__line][self.__caret_pos:])
        self.__cont[self.__line] = self.__cont[self.__line][:self.__caret_pos]
        self.__line += 1
        self.__caret_pos = 0

    def __add_char(self, char):
        """Add a character to the text."""
        lno, cpos = self.__line, self.__caret_pos
        before_caret = self.__cont[lno][:cpos]
        after_caret  = self.__cont[lno][cpos:]
        self.__cont[lno] = before_caret + char + after_caret
        self.__caret_pos += 1

    def __move_end(self):
        """Move the caret to the end of the last line."""
        self.__line = len(self.__cont) - 1
        self.__caret_pos = len(self.__cont[self.__line])

    def __move_home(self):
        """Move the caret to the beginning of the first line."""
        self.__line = 0
        self.__caret_pos = 0

    def __move_right(self):
        """Move the caret to the right."""
        if self.__caret_pos < len(self.__cont[self.__line]):
            self.__caret_pos += 1
        elif self.__line < len(self.__cont) - 1:
            self.__line += 1
            self.__caret_pos = 0

    def __move_left(self):
        """Move the caret to the left."""
        if self.__caret_pos > 0:
            self.__caret_pos -= 1
        elif self.__line > 0:
            self.__line -= 1
            self.__caret_pos = len(self.__cont[self.__line])

    def __move_right_word(self):
        """Move the caret to the right."""
        log.debug("moving right one word")
        if self.__caret_pos == len(self.__cont[self.__line]):
            if self.__line < len(self.__cont) - 1:
                self.__line += 1
                self.__caret_pos = 0
            else:
                return

        line = self.__cont[self.__line]
        while self.__caret_pos < len(line) and line[self.__caret_pos].isspace():
            self.__caret_pos += 1

        while self.__caret_pos < len(line) and not line[self.__caret_pos].isspace():
            self.__caret_pos += 1

    def __move_left_word(self):
        """Move the caret to the left."""
        log.debug("moving left one word")
        if self.__caret_pos == 0:
            if self.__line > 0:
                self.__line -= 1
                self.__caret_pos = len(self.__cont[self.__line])
            else:
                return
        while self.__caret_pos > 0 and self.__cont[self.__line][self.__caret_pos - 1].isspace():
            self.__caret_pos -= 1
        while self.__caret_pos > 0 and not self.__cont[self.__line][self.__caret_pos - 1].isspace():
            self.__caret_pos -= 1

    def __move_down(self):
        """Move the caret down."""
        if self.__line < len(self.__cont) - 1:
            self.__line += 1
            self.__caret_pos = min(self.__caret_pos, len(self.__cont[self.__line]))

    def __move_up(self):
        """Move the caret up."""
        if self.__line > 0:
            self.__line -= 1
            self.__caret_pos = min(self.__caret_pos, len(self.__cont[self.__line]))

    def move_caret(self, direction):
        """Move the caret in the text."""
        { "End":   self.__move_end,
          "Home":  self.__move_home,
          "Ctrl-Right": self.__move_right_word,
          "Ctrl-Left":  self.__move_left_word,
          "Right": self.__move_right,
          "Left":  self.__move_left,
          "Down":  self.__move_down,
          "Up":    self.__move_up }[direction]()

    def to_string(self):
        """Return the text as a string."""
        return "\n".join(self.__cont)

    def lines(self):
        """Return the text split by lines."""
        return self.__cont

    def strlen(self):
        """Return the length of the text."""
        return len(self.to_string())

    def caret_line(self, new_line = None):
        """Return the current line."""
        if new_line is not None:
            self.__line = new_line
        return self.__line

    def caret_pos(self, new_pos = None):
        """Return the caret position."""
        if new_pos is not None:
            self.__caret_pos = new_pos
        return self.__caret_pos

    def add_text(self, text):
        """Add text to the text."""
        # split text by newline
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if i == 0:
                self.__cont[self.__line] += line
                self.__caret_pos += len(text)
            else:
                self.__cont.insert(self.__line + i, line)
                self.__caret_pos = len(line)

    def set_text(self, text):
        """Set the text."""

        lines = text.split("\n")
        self.__cont = lines
        self.__line = len(lines) - 1
        self.__caret_pos = len(lines[-1])

    def update_by_key(self, keyname, char):
        """Update the text by key press."""
        if keyname == "BackSpace": # and cur["caret_pos"] > 0:
            self.__backspace()
        elif keyname in ["Home", "End", "Down", "Up", "Right", "Left", "Ctrl-Left", "Ctrl-Right"]:
            self.move_caret(keyname)
        elif keyname == "Return":
            self.__newline()
        elif char and char.isprintable():
            self.__add_char(char)
