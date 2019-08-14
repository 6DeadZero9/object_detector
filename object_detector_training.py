from functions import *

number_of_clusters = 2

"""
    Detector training using multiclustering
"""
#data_for_Kmeans = sorted(preparing_cluster_dataset("training.xml"))
#kmeans = Kmeans_clustering(data_for_Kmeans, number_of_clusters)
#XML_creating("training.xml", kmeans, number_of_clusters)
#create_detectors()

"""
   Detector training on a single cluster
"""
#aspect_ratio_dataset = sorted(preparing_cluster_dataset("training.xml"))
#bounding_box_correction(aspect_ratio_dataset, "training.xml")
"""
    Training image samples collection
"""
image_extraction()
