import tarfile
import os, sys
import urllib.request

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

def extract_now(source_path, destination_path):
    try:
        print("extracting ...")
        tarfile.open(source_path, 'r:gz').extractall(destination_path)
        return 1
    except:
        return 0



def extract_data_set(dir_name='cifar-10-batches-bin'):
    files_in_directory = os.listdir(os.getcwd())
    is_present = dir_name in files_in_directory
    if (is_present == False):
        try:
            """Download and extract the tarball from Alex's website."""
            print("running download and/or extract...")
            source_root_directory = os.getcwd()
            file_tar_gz = DATA_URL.split('/')[-1]
            source_file_tar_gz_path = os.path.join(source_root_directory, file_tar_gz)

            if(file_tar_gz in files_in_directory):
                return extract_now(source_path=source_file_tar_gz_path, destination_path=source_root_directory)

            else:
                def _progress(count, block_size, total_size):
                    sys.stdout.write('\r>> Downloading %s %.1f%%' % (file_tar_gz,
                                                                     float(count * block_size) / float(total_size) * 100.0))
                    sys.stdout.flush()

                source_file_tar_gz_path, _ = urllib.request.urlretrieve(DATA_URL, source_file_tar_gz_path,
                                                                        reporthook=_progress)
                stat_info = os.stat(source_file_tar_gz_path)
                print("\n")
                print('Successfully downloaded', file_tar_gz, stat_info.st_size, 'bytes.')
                return extract_now(source_path=source_file_tar_gz_path,destination_path=source_root_directory)
        except Exception as e:
            print("exception generated:", e)
            return 0
    else:
        return 1
