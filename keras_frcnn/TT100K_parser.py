import cv2
import numpy as np
import json


def get_data(input_path):
    # input_path: path to json file eg. '/home/ruiguo/TT100K/data/annotations.json'
    path_tmp = input_path.split('/')
    abs_data_path = '/'.join(path_tmp[:-1])
    found_bg = False
    all_imgs = {}

    classes_count = {}

    class_mapping = {}

    visualise = True

    with open(input_path, encoding='utf-8') as f:

        print('Parsing annotation files')

        annotation = json.load(f)
        imgs = annotation['imgs']
        for id in imgs:
            img = imgs[id]
            type = img['path'].split('/')[0]
            filename = abs_data_path + '/' + img['path']
            if filename not in all_imgs:
                if type == 'other':
                    continue
                all_imgs[filename] = {}
                if type == 'test':
                    all_imgs[filename]['imageset'] = 'test'
                if type == 'train':
                    all_imgs[filename]['imageset'] = 'trainval'
                im = cv2.imread(filename)
                (rows, cols) = im.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []
                if img['objects']:
                    for ele in img['objects']:
                        class_name = ele['category']
                        if class_name not in classes_count:
                            classes_count[class_name] = 1
                        else:
                            classes_count[class_name] += 1
                        if class_name not in class_mapping:
                            class_mapping[class_name] = len(class_mapping)
                        x1 = ele['bbox']['xmin']
                        x2 = ele['bbox']['xmax']
                        y1 = ele['bbox']['ymin']
                        y2 = ele['bbox']['ymax']
                        all_imgs[filename]['bboxes'].append(
                            {'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        return all_data, classes_count, class_mapping
