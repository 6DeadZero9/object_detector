import numpy as np
import pandas as pd
import cv2
import xml.etree.ElementTree as ET

def data_extraction(xmlfile):
	name = xmlfile.split('.')[0]
	csv_data = pd.DataFrame(columns=['FileName', 'XMin', 'XMax', 'YMin', 'YMax', 'ClassName'])
	tree = ET.parse(xmlfile)
	root = tree.getroot()
	for branch in root:		
		if branch.tag == 'images':
			for image in branch:
				for index, box in enumerate(image):	
					csv_data = csv_data.append({'FileName': image.attrib['file'], 
                                        'XMin': int(box.attrib['left']), 
                                        'XMax': int(box.attrib['left']) + int(box.attrib['width']), 
                                        'YMin': int(box.attrib['top']), 
                                        'YMax': int(box.attrib['top']) + int(box.attrib['height']), 
                                        'ClassName': 'sausage'}, 
                                       ignore_index=True)
	csv_data.to_csv('{}.csv'.format(name))
            
train = pd.read_csv('training.csv')
data = pd.DataFrame()
data['format'] = train['FileName']
for i in range(data.shape[0]):
	data['format'][i] = data['format'][i] + ',' + str(train['XMin'][i]) + ',' + str(train['YMin'][i]) + ',' + str(train['XMax'][i]) + ',' + str(train['YMax'][i]) + ',' + train['ClassName'][i]

data.to_csv('training.txt', header=None, index=None, sep=' ')	