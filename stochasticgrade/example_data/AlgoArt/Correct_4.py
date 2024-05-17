import math
import random
import numpy as np
import colorsys

ALPHA = 1.6
WIDTH = 600
HEIGHT = 600
N_CIRCLES = 5000

_LIST_OF_RADII = []

def main():
    for i in range(N_CIRCLES):
        r = random.paretovariate(ALPHA) - 1
        x = np.random.uniform(r, WIDTH-r)
        y = np.random.uniform(r, HEIGHT-r)
        if r < 150:
            draw_circle(r, x, y, random_color())
    return _LIST_OF_RADII

def random_color():
    """
    Generates a random color in HSV space, then
    translates it into RGB space. Feel free to edit
    """
    h = np.random.uniform(200,300) / 360
    s = np.random.uniform(0.5, 0.9)
    v = np.random.uniform(.7, .9)
    (r,g,b) = colorsys.hsv_to_rgb(h, s, v)
    return f'rgb({r*256},{g*256},{b*256})'

def draw_circle(radius, center_x, center_y, color):
    _LIST_OF_RADII.append(radius)

if __name__ == '__main__':
    main()