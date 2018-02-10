from PIL import Image, ImageDraw
import math

def getcolor(x, y):
    '''
    :param x: x-coordinate from center of circle
    :param y: y-coordinate from center of circle
    :return: tuple (R, G, B) calculated through a poly polynomial fit
    '''
    return getRGB((x, y), 'poly')

def getRGB(vector, fit):
    '''
    :param vector: normalized (x, y) relative to center of circle
    :param fit: Determines how vector maps to RGB. Can be 'lin', 'log', 'poly'
    :return: tuple (R, G, B) calculated through specified fit
    '''
    assert fit in ('lin', 'log', 'poly')
    color = []
    for c in ('r', 'g', 'b'):
        rotv = rotate(vector, c)
        if rotv[0] > - 0.95:
            if fit == 'lin':
                color.append(int(128 * (1 + rotv[0])))
            elif fit == 'log':
                color.append(int(256 * math.log2(((1 + rotv[0]) / 2) + 1)))
            elif fit == 'poly':
                color.append(int(90 * (1 + rotv[0]) ** (3 / 2)))
        else:
            color.append(0)
    return tuple(color)

def rotate(vector, color):
    '''
    :param vector: (x, y)
    :param color: 'r': no rotation; 'g': rotation by -120 deg; 'b': rotation by 120 deg
    :return: (x', y') of rotated vector
    '''
    if color == 'r':
        return vector
    elif color == 'g':
        return ((-0.5 * vector[0]) + ((math.sqrt(3) / 2) * vector[1]),
                (-0.5 * vector[1]) - ((math.sqrt(3) / 2) * vector[0]))
    elif color == 'b':
        return ((-0.5 * vector[0]) - ((math.sqrt(3) / 2) * vector[1]),
                (-0.5 * vector[1]) + ((math.sqrt(3) / 2) * vector[0]))
    else:
        raise ValueError('Invalid Color.')

def normalize(vector):
    mag = magnitude(vector)
    return (vector[0] / mag, vector[1] / mag)

def magnitude(vector):
    if vector == (0, 0):
        return 1
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2)

def generateColorWheel(fit='poly', size=2000):
    '''
    :param fit: Determine how each pixel maps to RGB. Can be 'lin', 'log', 'poly'
    :param size: Pixel length of the square box containing the wheel
    :return: Saves a png of a color wheel in the same directory.
    '''
    img_half = outer_radius = size // 2

    im = Image.new('RGB', (size, size))
    for x in range(size):
        for y in range(size):
            pix = (x, y)
            v = (x - img_half, y - img_half)
            if magnitude(v) > outer_radius:
                continue

            im.putpixel(pix, getRGB(normalize(v), fit))

    bigsize = (im.size[0] * 3, im.size[1] * 3)
    mask = Image.new('L', bigsize, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + bigsize, fill=255)
    mask = mask.resize(im.size, Image.ANTIALIAS)
    im.putalpha(mask)

    im.save('NewWheel_' + fit + '.png')