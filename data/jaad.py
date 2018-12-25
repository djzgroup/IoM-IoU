import os
import os.path
import torch
import torch.utils.data as data
import cv2, pickle
import numpy as np

CLASSES = (  # always index 0
        'cross', 'stop')

class AnnotationTransform(object):
    """
    Same as original
    Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of UCF24's 24 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(CLASSES, range(len(CLASSES))))
        self.ind_to_class = dict(zip(range(len(CLASSES)),CLASSES))

    def __call__(self, bboxs, width, height):
        res = []
        for t in range(len(bboxs)):
            bbox = bboxs[t,:]
            label = bboxs[t, 4]
            '''pts = ['xmin', 'ymin', 'xmax', 'ymax']'''
            bndbox = []
            for i in range(4):
                scale =  width if i % 2 == 0 else height
                cur_pt = min(scale, int(bbox[i]))
                cur_pt = float(cur_pt) / scale
                bndbox.append(cur_pt)
            bndbox.append(label)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

def make_lists(rootpath, imgtype, split=1, fulltest=False):
    with open(rootpath) as f:
        lines = f.readlines()
    np.random.shuffle(lines)
    return lines[int(0.1*len(lines)):], lines[:int(0.1*len(lines))], lines

class JAADDetection(data.Dataset):

    def __init__(self, root, image_set, transform=None, target_transform=None,
                 dataset_name='ucf24', input_type='rgb', full_test=False):
        self.input_type = input_type
        input_type = input_type+'-images'
        self.root = root
        self.CLASSES = CLASSES
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join(root, 'labels/', '%s.txt')
        self._imgpath = os.path.join(root, input_type)
        self.ids = list()

        trainlist, testlist, video_list = make_lists(root, input_type, split=1, fulltest=full_test)
        self.video_list = video_list
        if self.image_set == 'train':
            self.ids = trainlist
        elif self.image_set == 'test':
            self.ids = testlist
        else:
            print('spacify correct subset ')

    def __getitem__(self, index):
        im, gt, img_index = self.pull_item(index)

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        line = self.ids[index]
        line = line.split(' ')
        img = cv2.imread(line)
        height, width, channels = img.shape
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        target = self.target_transform(box, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        # print(height, width,target)
        return torch.from_numpy(img).permute(2, 0, 1), target, index

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """

    targets = []
    imgs = []
    image_ids = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        image_ids.append(sample[2])
    return torch.stack(imgs, 0), targets, image_ids