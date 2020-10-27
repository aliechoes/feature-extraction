import os
import glob
import logging
import pandas as pd
import numpy as np
from imageio import imread
import features
import segmentation

LOGING_LEVEL = 1

def logger(level):
    logging_level = {
        0 : logging.WARNING,
        1 : logging.INFO,
        2 : logging.DEBUG
    }
    logging.basicConfig(level=logging_level[level])
    logging.getLogger('matplotlib.font_manager').disabled = True

def finding_classes(data_dir):
    """
    this function finds the folders in the root path and considers them
    as classes
    """
    classes = sorted(os.listdir(data_dir))
    logging.info("Classes: %s \n" % classes)
    return classes


def finding_channels(classes, data_dir):
    """
    this function finds the existing channels in the folder and returns
    a list of them
    """
    channels = ["Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6", \
                               "Ch7", "Ch8", "Ch9", "Ch10", "Ch11","Ch12", \
                               "Ch13", "Ch14", "Ch15", "Ch16", "Ch17", "Ch18"]
    existing_channels = []
    for ch in channels:
        cl_path = os.path.join(data_dir, classes[0], "*_" +  ch + "*")
        cl_files = glob.glob(cl_path)
        if len(cl_files)> 1:
            existing_channels.append(ch)
    return existing_channels


def number_of_files_per_class(df ):
    """
    this function finds the number of files in each folder. It is important to
    consider that we consider all the channels togethr as on single image
    output: dictionary with keys as classes and values as number of separate images
    """

    logging.info("detected independent images per classes") 
    logging.info(df.groupby(["class", "set"])["class"].agg("count")) 
    
    return None
          

def run(data_dir):
    pass



if __name__ == "__main__":
    data_dir = "somewhere"
    run(data_dir)
