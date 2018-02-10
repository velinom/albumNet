from PIL import Image
from random import randint
import os

def random_image(width, height):
    pil_map = Image.new("RGBA", (width, height), 255)
    random_grid = [0] * width * height
    for i in range(width * height):
        random_grid[i] = (randint(0, 256), randint(0, 256), randint(0, 256))
    pil_map.putdata(random_grid)
    return pil_map

im = random_image(256, 256)
im.show()

folder_path = os.path.abspath(os.getcwd() + '/TestAlbums')

def filter_images():

    i = 1
    for subdir, dirs, files in os.walk(os.getcwd()):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith('.jpg'):
                process_image(filepath, i)
                os.remove(filepath)
                i += 1
                #print (filepath)
            elif filepath.endswith('.py'):
                pass
            else:
                os.remove(filepath)

def process_image(fp, i):
    try:
        im = Image.open(fp)
        im = im.resize((256, 256))
        im.save(os.path.abspath(folder_path + '/' + str(i) + '.jpg'))
    except OSError:
        return

#filter_images()
