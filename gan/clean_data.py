from PIL import Image
import os

folder_path='./data/'

def process_image(fp):
    try:
        im = Image.open(fp)
        im.resize((256, 256))
        #im.save(os.path.abspath(folder_path + '/' + fp))
    except OSError:
        print(fp)
        return


def files():
    current_dir = os.getcwd()
    albums_dir = os.path.join(current_dir, 'data')
    for each in os.listdir(albums_dir):
        process_image(folder_path + each)

if __name__ == "__main__":
    files()