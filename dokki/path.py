import os
import logging
import zipfile
import ntpath



def is_dir_not_empty(path):
    return (os.path.exists(path) and len(os.listdir(path))>0)     

def extract_tar(filename: str, directory_to_extract_to : str) -> None:
    if is_dir_not_empty(directory_to_extract_to):
        logging.info("Output path (%s) it not empty. Jar %s will not be extracted.",directory_to_extract_to, filename)
        return
    with tarfile.open(filename, "r") as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=directory_to_extract_to)


def extract_zip(filename: str, directory_to_extract_to : str) -> None:
    if is_dir_not_empty(directory_to_extract_to):
        logging.info("Output path (%s) it not empty. Zip %s will not be extracted.",directory_to_extract_to, filename)
        return
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

def extract_filename_from_path(path):
    return ntpath.basename(path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print( is_dir_not_empty("/usr") )
    extract_zip("/home/gugaime/Downloads/image.zip", "/tmp/myimg")