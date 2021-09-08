import wget
from zipfile import ZipFile
import argparse
import os


def argument_parser():
    """
    Construct an argument parser and return the arguments.

    :return: Arguments.
    """

    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-u", "--url", required=True,
                    help="url to dataset", type=str)
    return vars(ap.parse_args())


def main():
    """
    Provide the URL to a dataset.
    The dataset will be downloaded to "./Datasets/" directory.
    The dataset is then extracted.
    """

    args = argument_parser()

    site_url = args['url']
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
