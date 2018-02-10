from PIL import Image
from collections import namedtuple
from math import sqrt
import random

# inspired by http://charlesleifer.com/blog/using-python-and-k-means-to-find-the-dominant-colors-in-images/

NUM_PTS = 3

ColorPoint = namedtuple('ColorPoint', ('rgb', 'count'))
Cluster = namedtuple('Cluster', ('points', 'center'))

def getColorPoints(image):
    colors = []
    w, h = image.size
    for count, rgb in image.getcolors(w * h):
        colors.append(ColorPoint(rgb, count))
    return colors

def euclidean(cp1, cp2):
    sum = 0
    for i in range(NUM_PTS):
        sum += (cp1.rgb[i] - cp2.rgb[i]) ** 2
    return sqrt(sum)

def calculate_center(points):
    vals = [0.0 for i in range(NUM_PTS)]
    plen = 0
    for p in points:
        plen += p.count
        for i in range(NUM_PTS):
            vals[i] += (p.rgb[i] * p.count)
    return ColorPoint([(v / plen) for v in vals], 1)

def kmeans(colorPoints, min_diff):
    clusters = [Cluster([col], col) for col in random.sample(colorPoints, NUM_PTS)]

    while 1:
        # generate NUM_PTS lists by clustering colors together
        ColLists = [[] for i in range(NUM_PTS)]

        # group each point with its closest cluster center
        for cp in colorPoints:
            smallest_distance = float('Inf')
            for i in range(NUM_PTS):
                distance = euclidean(cp, clusters[i].center)
                if distance < smallest_distance:
                    smallest_distance = distance
                    idx = i
            ColLists[idx].append(cp)

        diff = 0
        for i in range(NUM_PTS):
            old = clusters[i]
            center = calculate_center(ColLists[i])
            new = Cluster(ColLists[i], center)
            clusters[i] = new
            diff = max(diff, euclidean(old.center, new.center))

        if diff < min_diff:
            break

    return clusters

def colorz(filename):
    img = Image.open(filename)
    img.thumbnail((200, 200))

    points = getColorPoints(img)
    clusters = kmeans(points, 1)
    finalColors = []
    for c in clusters:
        finalColors.append(c.center.rgb)
    return finalColors