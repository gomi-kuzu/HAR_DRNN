#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""

"""

import zipfile
import os
import shutil
import urllib

def unzip(filename, path='.'):
    with zipfile.ZipFile(filename, 'r') as zip_file:
        zip_file.extractall(path=path)

if __name__ == "__main__":
    fd = os.path.abspath(os.path.dirname(__file__))
    url = "http://bit.ly/i0ivEz"
    urllib.urlretrieve(url,"{}/downloaded".format(fd))
    unzip("{}/downloaded".format(fd),fd)
    shutil.copytree("{}/HascToolDataPrj/SampleData/0_sequence".format(fd), "{}/HascToolDataPrj/SampleData_sequence".format(fd))
    shutil.rmtree("{}/HascToolDataPrj/SampleData/0_sequence".format(fd))
    os.rename("{}/HascToolDataPrj/SampleData".format(fd), "{}/HascToolDataPrj/SampleData_non_sequence".format(fd))
    os.remove("{}/downloaded".format(fd))
    os.remove("{}/HascToolDataPrj/.project".format(fd))
    shutil.rmtree("{}/HascToolDataPrj/HASCXBD".format(fd))
    shutil.rmtree("{}/HascToolDataPrj/temp".format(fd))
    shutil.rmtree("{}/HascToolDataPrj/.settings".format(fd))

    print("Downloading completion")
