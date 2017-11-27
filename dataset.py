import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np
import imutils

def rotateImage(image, image_size, angle):
  image_center = tuple(np.array( (image_size, image_size))/2)
  rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
  result = cv2.warpAffine(image, rot_mat, (image_size, image_size) ,
    None, flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

  return result

def load_train(train_path, image_size, classes):
    import ml_helper as ml

    images = []
    labels = []
    img_names = []
    cls = []

    print('Reading Training images from {}'.format(train_path))
    for fields in classes:
        index = classes.index(fields)
        path = os.path.join(train_path, fields, '*g')
        files = glob.glob(path)
        print('Index {} - Path {} - Files {}'.format(index, fields, len(files)))


        print('\r')
        ml.printProgressBar(0, len(files), prefix = 'Progress:', suffix = 'Complete', length = 50)

        t = 0
        for fl in files:

            # uncomment for 8bpp
            #image = cv2.imread(fl,0)
            image = cv2.imread(fl)


            image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)

            label = np.zeros(len(classes))
            label[index] = 1.0

            flbase = os.path.basename(fl)

            for angle in np.arange(0, 360, 5):
                rotated = rotateImage(image, image_size, angle)

                flbase += 'R' + str(angle) + '_'

                for x in range(2, 8):
                    frame_darker = (image * (x/6)).astype(np.float32)

                    images.append(frame_darker)
                    labels.append(label)
                    flbase += 'B' + str(x)

                    img_names.append(flbase)
                    cls.append(fields)

            ml.printProgressBar(t, len(files), prefix = 'Progress:', suffix = 'Complete', length = 50)
            t += 1

    print('Complete Images Array :' + str(len(images)))

    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls


class DataSet(object):

  def __init__(self, images, labels, img_names, cls):
    self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._img_names = img_names
    self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def img_names(self):
    return self._img_names

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, validation_size):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, labels, img_names, cls = load_train(train_path, image_size, classes)
  images, labels, img_names, cls = shuffle(images, labels, img_names, cls)

  if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])

  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  validation_img_names = img_names[:validation_size]
  validation_cls = cls[:validation_size]

  train_images = images[validation_size:]
  train_labels = labels[validation_size:]
  train_img_names = img_names[validation_size:]
  train_cls = cls[validation_size:]

  data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

  return data_sets
