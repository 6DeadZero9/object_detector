import os
import sys
import glob
import dlib
import re
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def parseXML_v1(xmlfile):
    """
        Function is not used
    """
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    images = []
    boxes = []
    for branch in root:
        if branch.tag == 'images':
            for image in branch:
                current_image = cv2.imread(image.attrib['file'])
                images.append(current_image)
                current_boxes = ([],)
                for index, box in enumerate(image):
                    current_boxes[0].append(dlib.rectangle(left = int(box.attrib['left']), top = int(box.attrib['top']), right = int(box.attrib['left']) + int(box.attrib['width']), bottom = int(box.attrib['top']) + int(box.attrib['height'])))
                boxes.append(current_boxes)
    return images, boxes

def preparing_cluster_dataset(xmlfile):
    """
        This function is used for collecting aspect ratio of every bounding box in training images and making a dataset out of this data, that will be passed into the Kmeans_clustering(*args) function
        
        Args:
            - xmlfile: main training.xml file that will be parsed in process
        Returns:
            - dataset: collected aspect ratio dataset
    """
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    dataset = []
    for branch in root:
        if branch.tag == 'images':
            for image in branch:
                current_image = cv2.imread(image.attrib['file'])
                average = 0
                current_averages = []
                print("\n" + image.attrib['file'])
                for index, box in enumerate(image):
                    aspect_ration = round(int(box.attrib['width'])/int(box.attrib['height']), 2)
                    average += aspect_ration
                    current_averages.append(aspect_ration)
                    print('{} box sides aspect ratio: '.format(index), aspect_ration)
                dataset += current_averages
    return dataset

def Kmeans_clustering(dataset, number_of_clusters):
    """
        This function is used to train a model based on given dataset using kmeans algorithm
        
        Args:
        - dataset: training dataset collected from given training.xml in preparing_cluster_dataset(*args) function
        - number_of_clusters: the name is self-explenatory
        Returns:
        - kmeans: trained model
        """
    dataset_copy = np.asarray(dataset)
    kmeans = KMeans(n_clusters = number_of_clusters, random_state = 0).fit(dataset_copy.reshape(-1,1))
    return kmeans

def XML_creating(xmlfile, kmeans, number_of_clusters):
    """
        This function is used for creating new xml training files divided into groups(using kmeans algorithm).
        Firstly it parses the main training.xml file to group up the training data into separate classes.
        Secondly the function creates new xml training files and stores them in the xml folder
        
        Args:
        - xmlfile: main training.xml file that will be parsed in process
        - kmeans: trained model for group clustering
        - number_of_clusters: the name is self-explenatory
        Returns:
        None
        """
    dataset_to_convert = {cluster_number: {} for cluster_number in range(number_of_clusters)}
    main_tree = ET.parse(xmlfile)
    main_root = main_tree.getroot()
    for branch in main_root:
        if branch.tag == 'images':
            for image in branch:
                for box in image:
                    current_image = cv2.imread(image.attrib['file'])
                    aspect_ration = np.asarray([round(int(box.attrib['width'])/int(box.attrib['height']), 2)],)
                    kmeans_prediction = kmeans.predict(aspect_ration.reshape(-1, 1))
                    if 'images' not in dataset_to_convert[kmeans_prediction[0]].keys():
                        dataset_to_convert[kmeans_prediction[0]]['images'] = {}
                    if image.attrib['file'] not in dataset_to_convert[kmeans_prediction[0]]['images'].keys():
                        dataset_to_convert[kmeans_prediction[0]]['images'][image.attrib['file']] = []
                    dataset_to_convert[kmeans_prediction[0]]['images'][image.attrib['file']].append(box.attrib)
    for cluster in dataset_to_convert.keys():
        dataset = ET.Element('dataset')
        name = ET.SubElement(dataset, 'name')
        name.text = 'imglab dataset'
        comment = ET.SubElement(dataset, 'comment')
        comment.text = 'Created by imglab tool.'
        images = ET.SubElement(dataset, 'images')
        for image in dataset_to_convert[cluster]['images'].keys():
            current_image = ET.SubElement(images, 'image', { 'file': '{}'.format('../' + image) })
            for box in dataset_to_convert[cluster]['images'][image]:
                current_box = ET.SubElement(current_image, 'box', box)
                label = ET.SubElement(current_box, 'label')
                label.text = 'sausage'
        rough_string = ET.tostring(dataset, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        reparsed = reparsed.toprettyxml(indent="  ")
        myfile = open("xml/group_{}.xml".format(cluster), "w")
        myfile.write(reparsed)


def create_detectors():
    """
        This function is used for creating shape detectors using newly created training xml files by XML_creating(*args) function
        
        Args:
            None
        Returns:
            None
    """
    path_to_xml = os.listdir('xml')
    options = dlib.simple_object_detector_training_options()
    options.add_left_right_image_flips = True
    options.C = 4
    options.num_threads = 4
    options.be_verbose = True
    for current_xml in path_to_xml:
        if '.xml' in current_xml:
            splited = re.split('[_.]' ,current_xml)
            current_cluster = int(splited[1])
            dlib.train_simple_object_detector('xml/' + current_xml, "detectors/detector_{}.svm".format(current_cluster), options)
            print("Training accuracy: {}".format(dlib.test_simple_object_detector('xml/' + current_xml, "detectors/detector_{}.svm".format(current_cluster))))
            print("Testing accuracy: {}".format(dlib.test_simple_object_detector("testing.xml", "detectors/detector_{}.svm".format(current_cluster))))

