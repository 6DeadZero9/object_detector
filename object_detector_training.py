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
#options = dlib.simple_object_detector_training_options()
#options.add_left_right_image_flips = True
#options.C = 4
#options.num_threads = 4
#options.be_verbose = True
#dlib.train_simple_object_detector("training.xml", "detectors/detector.svm", options)
#print("Training accuracy: {}".format(dlib.test_simple_object_detector("training.xml", "detectors/detector.svm")))
#print("Testing accuracy: {}".format(dlib.test_simple_object_detector("testing.xml", "detectors/detector.svm")))
"""
    Training image samples collection
"""
image_extraction()
"""
    Correction of the bounding bos aspect ratio
"""
#aspect_ratio_dataset = sorted(preparing_cluster_dataset("training.xml"))
#bounding_box_correction(aspect_ratio_dataset, "training.xml")
