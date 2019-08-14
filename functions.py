import os
import glob
import dlib
import re
import numpy as np
import cv2
import json
import wget
import keyboard
import xml.etree.ElementTree as ET
from xml.dom import minidom
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from elasticsearch import Elasticsearch

def bounding_box_correction(dataset, xmlfile):
    averege = round(sum(dataset) / len(dataset), 2)
    procent_step = 0.25
    bottom_value, top_value = round(averege - averege * procent_step, 2), round(averege + averege * procent_step, 2)
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    for branch in root:
        if branch.tag == 'images':
            for image in branch:
                for index, box in enumerate(image):
                    this_box_aspect_ratio = round(int(box.attrib['width']) / int(box.attrib['height']), 2)
                    if not bottom_value <= this_box_aspect_ratio <= top_value:
                        box_side = 'right'
                        top, left, right, bottom = int(box.attrib['top']), int(box.attrib['left']), int(box.attrib['left']) + int(box.attrib['width']), int(box.attrib['top']) + int(box.attrib['height'])
                        this_box_aspect_ratio = round(int(box.attrib['width']) / int(box.attrib['height']), 2)
                        old_stats = {'aspect_ratio': this_box_aspect_ratio, 'box_side': box_side}
                        while True:
                            current_stats = {'aspect_ratio': round((right - left) / (bottom - top), 2), 'box_side': box_side}
                            current_image = cv2.imread(image.attrib['file'])
                            if keyboard.is_pressed('up'):
                                box_side = 'top'
                            elif keyboard.is_pressed('down'):
                                box_side = 'bottom'
                            elif keyboard.is_pressed('left'):
                                box_side = 'left'
                            elif keyboard.is_pressed('right'):
                                box_side = 'right'
                            if keyboard.is_pressed('-'):
                                if box_side == 'top':
                                    top -= 1
                                elif box_side == 'bottom':
                                    bottom -= 1
                                elif box_side == 'left':
                                    left -= 1
                                elif box_side == 'right':
                                    right -= 1
                            elif keyboard.is_pressed('='):
                                if box_side == 'top':
                                    top += 1
                                elif box_side == 'bottom':
                                    bottom += 1
                                elif box_side == 'left':
                                    left += 1
                                elif box_side == 'right':
                                    right += 1
                            if current_stats != old_stats:
                                old_stats = current_stats
                                print("\nCurrent box side: {0}\nCurrent aspect ratio: {1}\nBottom value: {2}\nTop value: {3}".format(current_stats['box_side'], current_stats['aspect_ratio'], bottom_value, top_value))
                            cv2.rectangle(current_image, (left, top), (right, bottom), (0, 0, 255), 1)
                            cv2.imshow('Current image', current_image)
                            if cv2.waitKey(1) & 0xff == 27:
                                box.attrib['left'] = str(left)
                                box.attrib['top'] = str(top)
                                box.attrib['width'] = str(right - left)
                                box.attrib['height'] = str(bottom - top)
                                cv2.destroyWindow('Current image')
                                tree.write(xmlfile)
                                break

def image_extraction():
    """
        This function is used for collecting training images for object detector from elasticsearch.
        
        Args:
            None
        Returns:
            None
        """
    with open('es_data.json') as json_file:
        data = json.load(json_file)
    es = Elasticsearch(data['elasticsearch'])
    list_of_records = []
    single_scroll = 100
    for expo_id in data['expo_id']:
        search = es.search(index = data['database'], size = single_scroll, request_timeout = 60, body = {"query": {"bool": {"must": [{"match": {"expo_id": expo_id}}]}}}, scroll = '1m')
        scroll_size = search['hits']['total']
        if_scroll_size = scroll_size <= single_scroll
        scrollId = search['_scroll_id']
        for hit in search['hits']['hits']:
            list_of_records.append(hit)
        if not if_scroll_size:
            while scroll_size > 0:
                search = es.scroll(scroll_id = scrollId, scroll = '1m')
                scrollId = search['_scroll_id']
                scroll_size = len(search['hits']['hits'])
                for hit in search['hits']['hits']:
                    list_of_records.append(hit)
    for image in list_of_records:
        url = image['_source']['url']
        image_name = url.split('/')[-1]
        if image_name not in os.listdir('to_be_added'):
            current_image = wget.download(url, 'to_be_added')

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
                    aspect_ration = round(
                        int(box.attrib['width']) / int(box.attrib['height']), 2)
                    average += aspect_ration
                    current_averages.append(aspect_ration)
                    print('{} box sides aspect ratio: '.format(
                        index), aspect_ration)
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
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(
        dataset_copy.reshape(-1, 1))
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
    dataset_to_convert = {cluster_number: {}
                          for cluster_number in range(number_of_clusters)}
    main_tree = ET.parse(xmlfile)
    main_root = main_tree.getroot()
    for branch in main_root:
        if branch.tag == 'images':
            for image in branch:
                for box in image:
                    current_image = cv2.imread(image.attrib['file'])
                    aspect_ration = np.asarray(
                        [round(int(box.attrib['width']) / int(box.attrib['height']), 2)],)
                    kmeans_prediction = kmeans.predict(
                        aspect_ration.reshape(-1, 1))
                    if 'images' not in dataset_to_convert[kmeans_prediction[0]].keys(
                    ):
                        dataset_to_convert[kmeans_prediction[0]]['images'] = {}
                    if image.attrib['file'] not in dataset_to_convert[kmeans_prediction[0]]['images'].keys(
                    ):
                        dataset_to_convert[kmeans_prediction[0]
                                           ]['images'][image.attrib['file']] = []
                    dataset_to_convert[kmeans_prediction[0]]['images'][image.attrib['file']].append(
                        box.attrib)
    for cluster in dataset_to_convert.keys():
        dataset = ET.Element('dataset')
        name = ET.SubElement(dataset, 'name')
        name.text = 'imglab dataset'
        comment = ET.SubElement(dataset, 'comment')
        comment.text = 'Created by imglab tool.'
        images = ET.SubElement(dataset, 'images')
        for image in dataset_to_convert[cluster]['images'].keys():
            current_image = ET.SubElement(
                images, 'image', {'file': '{}'.format('../' + image)})
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
            splited = re.split('[_.]', current_xml)
            current_cluster = int(splited[1])
            if "detector_{}.svm".format(current_cluster) not in os.listdir("detectors"):
                dlib.train_simple_object_detector(
                    'xml/' + current_xml,
                    "detectors/detector_{}.svm".format(current_cluster),
                    options)
                print(
                    "Training accuracy: {}".format(
                        dlib.test_simple_object_detector(
                            'xml/' + current_xml,
                            "detectors/detector_{}.svm".format(current_cluster))))
                print(
                    "Testing accuracy: {}".format(
                        dlib.test_simple_object_detector(
                            "testing.xml",
                            "detectors/detector_{}.svm".format(current_cluster))))
