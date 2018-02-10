from PIL import Image, ImageDraw
import math

img_size = 2000
img_half = img_size // 2
outer_radius = 1000

def getcolor(x, y):
    return getRGB((x, y), '3/2')

def getRGB(vector, fit):
    assert fit in ('lin', 'log', '3/2')
    color = []
    for c in ('r', 'g', 'b'):
        rotv = rotate(vector, c)
        if rotv[0] > - (outer_radius * 0.95):
            if fit == 'lin':
                color.append(int(128 * (1 + (rotv[0] / outer_radius))))
            elif fit == 'log':
                color.append(int(256 * math.log2(((1 + (rotv[0] / outer_radius)) / 2) + 1)))
            elif fit == '3/2':
                color.append(int(90 * (1 + (rotv[0] / outer_radius)) ** (3 / 2)))
        else:
            color.append(0)
    return tuple(color)

def rotate(vector, color):
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

def generateColorWheel(fit):

    im = Image.new('RGB', (img_size, img_size))
    for x in range(img_size):
        for y in range(img_size):
            pix = (x, y)
            v = (x - img_half, y - img_half)
            if magnitude(v) > outer_radius:
                continue

            im.putpixel(pix, getRGB(v, fit))

    bigsize = (im.size[0] * 3, im.size[1] * 3)
    mask = Image.new('L', bigsize, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + bigsize, fill=255)
    mask = mask.resize(im.size, Image.ANTIALIAS)
    im.putalpha(mask)

    im.save('NewWheel_' + fit + '.png')