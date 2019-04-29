import numpy as np
import os
from xml.etree import ElementTree
import pickle
# Todo: Doc and test

class VOC2Pickle(object):

    def __init__(self, xml_path,class_list = ["Helmet","NoHelmet"]):
        self.path_prefix = xml_path
        self.num_classes = len(class_list)
        self.class_list = class_list
        self.data = dict()
        self._preprocess_XML()

    def _preprocess_XML(self):
        filenames = os.listdir(self.path_prefix)
        for filename in filenames:
            tree = ElementTree.parse(self.path_prefix + filename)
            root = tree.getroot()
            bounding_boxes = []
            one_hot_classes = []
            size_tree = root.find('size')
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)
            for object_tree in root.findall('object'):
                for bounding_box in object_tree.iter('bndbox'):
                    xmin = float(bounding_box.find('xmin').text)/width
                    ymin = float(bounding_box.find('ymin').text)/height
                    xmax = float(bounding_box.find('xmax').text)/width
                    ymax = float(bounding_box.find('ymax').text)/height
                bounding_box = [xmin,ymin,xmax,ymax]
                bounding_boxes.append(bounding_box)
                class_name = object_tree.find('name').text
                one_hot_class = self._to_one_hot(class_name)
                one_hot_classes.append(one_hot_class)
            image_name = root.find('filename').text
            bounding_boxes = np.asarray(bounding_boxes)
            one_hot_classes = np.asarray(one_hot_classes)
            image_data = np.hstack((bounding_boxes, one_hot_classes))
            self.data[image_name] = image_data

    def _to_one_hot(self, name):
        one_hot_vector = [0] * self.num_classes
        try:
            one_hot_vector[self.class_list.index(name)]
        except ValueError:
            print("Not know: ", name)

        return one_hot_vector
    def print(self):
        print(self.data)

    def save(self, filename):
        pickle.dump(self.data, open(filename, 'wb'))



## example on how to use it

# data = XML_preprocessor('./path2yourXMLFiles').data
# pickle.dump(data,open('./voc_2007.pkl','wb'))