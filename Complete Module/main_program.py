import os, shutil
from convert_jpeg import convert_jpeg
from image_preprocess import image_preprocess
from write_csv import append_list_as_row
from receipt_classification import receipt_classification
from pathlib import Path


def main():
    path = Path(__file__).parent.parent.absolute()

    # Directory for Raw File Folder
    raw = str(path) + r'\Dataset\raw'

    # Directory for proceeded File Folder
    proceeded = str(path) + r'\Dataset\proceeded'

    # Redirect to raw directory
    os.chdir(raw)

    # Standardized file format to JPEG
    convert_jpeg(raw, proceeded)

    # Redirect to proceeded directory
    os.chdir(proceeded)

    # Clean existing result csv file
    try:
        os.remove("output.csv")
    except:
        print("File has been removed")

    # Print header for result csv file
    row_contents = ['Filename', 'Issuer', 'Issue Date', 'Item List', 'Total Price', 'Keyword','label']
    append_list_as_row('output.csv', row_contents)

    # Image preprocessing, Extract text from
    image_preprocess(proceeded)
    receipt_classification()

if __name__ == "__main__":
    main()
