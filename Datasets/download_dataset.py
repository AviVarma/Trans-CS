import wget
from zipfile import ZipFile
import os

def main():
    site_url = "http://www.phontron.com/download/conala-corpus-v1.1.zip"
    wget.download(site_url)
    file = os.path.basename(site_url)

    with ZipFile(file, 'r') as zip:
        zip.printdir()

        print("extracting...")
        zip.extractall()
        print("Done!")


if __name__ == '__main__':
    main()
