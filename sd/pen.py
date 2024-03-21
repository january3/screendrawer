class Pen:
    """
    Represents a pen with customizable drawing properties.

    This class encapsulates properties like color, line width, and font settings
    that can be applied to drawing operations on a canvas.

    Attributes:
        color (tuple): The RGB color of the pen as a tuple (r, g, b), with each component ranging from 0 to 1.
        line_width (float): The width of the lines drawn by the pen.
        font_size (int): The size of the font when drawing text.
        fill_color (tuple or None): The RGB fill color for shapes. `None` means no fill.
        transparency (float): The transparency level of the pen's color, where 1 is opaque and 0 is fully transparent.
        font_family (str): The name of the font family used for drawing text.
        font_weight (str): The weight of the font ('normal', 'bold', etc.).
        font_style (str): The style of the font ('normal', 'italic', etc.).

    Args:
        color (tuple, optional): Initial color of the pen. Defaults to (0, 0, 0) for black.
        line_width (int, optional): Initial line width. Defaults to 12.
        transparency (float, optional): Initial transparency level. Defaults to 1 (opaque).
        fill_color (tuple, optional): Initial fill color. Defaults to None.
        font_size (int, optional): Initial font size. Defaults to 12.
        font_family (str, optional): Initial font family. Defaults to "Sans".
        font_weight (str, optional): Initial font weight. Defaults to "normal".
        font_style (str, optional): Initial font style. Defaults to "normal".

    Example usage:
        >>> my_pen = Pen(color=(1, 0, 0), line_width=5, transparency=0.5)
        >>> print(my_pen.color)
        (1, 0, 0)

    Note:
        This class does not directly handle drawing operations. It is used to store
        and manage drawing properties that can be applied by a drawing context.
    """

    def __init__(self, color = (0, 0, 0), line_width = 12, transparency = 1, fill_color = None, 
                 font_size = 12, font_family = "Sans", font_weight = "normal", font_style = "normal"):
        """
        Initializes a new Pen object with the specified drawing properties.
        """
        self.color        = color
        self.line_width   = line_width
        self.font_size    = font_size
        self.fill_color   = fill_color
        self.transparency = transparency
        #self.font_family       = font_family or "Segoe Script"
        self.font_family       = font_family or "Sans"
        self.font_weight       = font_weight or "normal"
        self.font_style        = font_style  or "normal"

    def color_set(self, color):
        self.color = color

    def font_set_from_description(self, font_description):
        self.font_family = font_description.get_family()
        self.font_size   = font_description.get_size() / Pango.SCALE
        self.font_weight = "bold"   if font_description.get_weight() == Pango.Weight.BOLD  else "normal"
        self.font_style  = "italic" if font_description.get_style()  == Pango.Style.ITALIC else "normal"

        print("setting font to", self.font_family, self.font_size, self.font_weight, self.font_style)


    def transparency_set(self, transparency):
        self.transparency = transparency
    
    def fill_set(self, color):
        self.fill_color = color

    def stroke_change(self, direction):
        # for thin lines, a fine tuned change of line width
        if self.line_width > 2:
            self.line_width += direction
        else:
            self.line_width += direction / 10
        self.line_width = max(0.1, self.line_width)

    def to_dict(self):
        return {
            "color": self.color,
            "line_width": self.line_width,
            "transparency": self.transparency,
            "fill_color": self.fill_color,
            "font_size": self.font_size,
            "font_family": self.font_family,
            "font_weight": self.font_weight,
            "font_style": self.font_style
        }

    def copy(self):
        return Pen(self.color, self.line_width, self.transparency, self.fill_color, self.font_size, self.font_family, self.font_weight, self.font_style)

    @classmethod
    def from_dict(cls, d):
        #def __init__(self, color = (0, 0, 0), line_width = 12, font_size = 12, transparency = 1, fill_color = None, family = "Sans", weight = "normal", style = "normal"):
        return cls(d.get("color"), d.get("line_width"), d.get("transparency"), d.get("fill_color"),
                   d.get("font_size"), d.get("font_family"), d.get("font_weight"), d.get("font_style"))


