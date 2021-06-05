import pandas as pd
from urllib.request import urlretrieve
from urllib.error import URLError
import os
from http.client import RemoteDisconnected, InvalidURL

prefix = '../data/pubfig/'


def download_images(filename):
    og_path = prefix + 'original_files/' + filename
    ds_path = prefix + 'dataset/'
    # assumption: the _urls files were manipulated s.t. they start with 'person	imagenum	url	rect	md5sum'
    df = pd.read_csv(og_path, sep='\t', names=['person', 'imagenum', 'url', 'rect', 'md5sum'])  # read to dataframe
    df = df.drop([0])  # remove redundant first line
    success_counter = 0

    for i in range(len(df)):
        # get necessary info and paths
        subdir_name = '_'.join(df['person'][i+1].split())  # compose sub-directory name for image dataset
        subdir_path = ds_path + subdir_name
        dl_url = df['url'][i+1]
        if not os.path.exists(subdir_path):  # create class directory if not there
            os.makedirs(subdir_path)

        try:
            file_name = dl_url.split('/')[-1]  # the actual filename
            write_path = ds_path + subdir_name + '/' + file_name
            urlretrieve(dl_url, write_path)  # download image to correct path
            success_counter += 1
            print('.')
        except (URLError, RemoteDisconnected, InvalidURL):  # not available or another error
            print(f'File @ {dl_url} could not be downloaded.')

    print(f'In total {success_counter} out of {len(df)} images could be downloaded.')



if __name__ == '__main__':
    download_images('dev_urls.txt')