import os
import sys
import glob
import dlib
import re
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from functions import *

number_of_clusters = 3

data_for_Kmeans = sorted(preparing_cluster_dataset("training.xml"))
kmeans = Kmeans_clustering(data_for_Kmeans, number_of_clusters)
XML_creating("training.xml", kmeans, number_of_clusters)
create_detectors()
