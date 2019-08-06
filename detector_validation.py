import os
import dlib
import numpy as np
import cv2

detectors = [dlib.fhog_object_detector('detectors/' + detector) for detector in os.listdir('detectors') if '.svm' in detector]
test_photos_directory = os.listdir('test')

for photo in test_photos_directory:
    if '.jpg' in photo or '.JPG' in photo:
        image = cv2.imread('test/' + photo)
        boxes, _, _ = dlib.fhog_object_detector.run_multiple(detectors, image, upsample_num_times = 1, adjust_threshold = 0.0)
        if boxes:
            for box in boxes:
                cv2.rectangle(image, (box.tl_corner().x, box.tl_corner().y), (box.br_corner().x, box.br_corner().y), (0, 0, 255), 2)
            cv2.imshow('{}'.format(photo), cv2.resize(image, (640, 400)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
