import os, shutil
from pdf2image import convert_from_path
from PIL import Image


def convert_jpeg(directory,output):
    # define image's path
    line_no = 0
    for filename in os.listdir(directory):
        # Convert and rename JPG file
        if filename.endswith(".jpg"):
            fname = filename.rsplit('.', 1)[0] + '.jpg'
            shutil.copy(src=directory + "/" + filename, dst=output + "/" + fname)

        # Convert and rename PDF file
        if filename.endswith(".pdf"):
            images = convert_from_path(filename, 300, poppler_path=r'C:\Program Files\poppler-0.68.0\bin')
            for img in images:
                fname = filename.rsplit('.', 1)[0] + '.jpg'
                img.save(output + "/" + fname, 'JPEG')
                break

        # Convert and rename PNG file
        if filename.endswith(".png"):
            im = Image.open(filename)
            rgb_im = im.convert('RGB')
            fname = filename.rsplit('.', 1)[0] + '.jpg'
            rgb_im.save(output + "/" + fname, 'JPEG')

