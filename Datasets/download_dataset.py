import wget
from zipfile import ZipFile
import os

def main():
    site_url = "http://www.phontron.com/download/conala-corpus-v1.1.zip"
    dataset_path = "./Datasets/"
    wget.download(site_url, dataset_path)
    file = os.path.basename(site_url)

    with ZipFile((dataset_path+file), 'r') as zip:
        zip.printdir()

        print("extracting...")
        zip.extractall(path=dataset_path)
        print("Done!")


if __name__ == '__main__':
    main()
