import os
import logging
import zipfile


def is_dir_not_empty(path):
    return (os.path.exists(path) and len(os.listdir(path))>0)     

def extract_tar(filename: str, directory_to_extract_to : str) -> None:
    if is_dir_not_empty(directory_to_extract_to):
        logging.info("Output path (%s) it not empty. Jar %s will not be extracted.",directory_to_extract_to, filename)
        return
    with tarfile.open(filename, "r") as tar:
        tar.extractall(path=directory_to_extract_to)


def extract_zip(filename: str, directory_to_extract_to : str) -> None:
    if is_dir_not_empty(directory_to_extract_to):
        logging.info("Output path (%s) it not empty. Zip %s will not be extracted.",directory_to_extract_to, filename)
        return
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print( is_dir_not_empty("/usr") )
    extract_zip("/home/gugaime/Downloads/image.zip", "/tmp/myimg")