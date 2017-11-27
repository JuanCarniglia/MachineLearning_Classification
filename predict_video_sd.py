# Inport Tensorflow
import tensorflow as tf

# Import Numpy
import numpy as np

# Import os
import os

# we need matplotlib to plot (this should go in a function)
import matplotlib
import matplotlib.pyplot as plt

# Disable boring warning messages of which there are so many...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '9'

# glob to read files recursively
import glob

# Open CV - For loading and Augmentation
from cv2 import *

# sys and argparse for ... args parsing!
import sys
import argparse

# helper for pretty printing with colors
import ml_helper as mlh

# imutils for image manipulation
import imutils

# small function to replace switch
def switch(x):
    return {
        0: 0,
        1: 0,
        2: 1,
        3: 1,
        4: 2,
        5: 2,
        6: 3,
        7: 3,
        8: 4,
        9: 4}[x]

# classes
classes = ['cincuenta_a', 'cincuenta_b', 'diez_a', 'diez_b','dos_pesos_a', 'dos_pesos_b', 'un_peso_a','un_peso_b', 'veinticinco_a', 'veinticinco_b']

# classes pretty text
classesTxt = ['Cincuenta Centavos', 'Diez Centavos', 'Dos Pesos',
              'Un Peso', 'Veinticinco Centavos']

# Path where image should be read from ... although here we are using video
dir_path = os.path.dirname(os.path.realpath(__file__))

# image expected size and BPP
image_size = 128
num_channels = 3

# some variables for debugging
debug = False
printCamStats = False
plotting = False

# Restore the saved model
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph(os.getcwd() + os.sep + 'models' + os.sep + 'm1' + os.sep + 'model.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint(os.getcwd() + os.sep + 'models' + os.sep + 'm1' + os.sep))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of
# the network
y_pred = graph.get_tensor_by_name("y_pred:0")

# Let's feed the images to the input placeholders
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")

y_test_images = np.zeros((1, len(classes)))

# init video capture
cap = cv2.VideoCapture(0)
width = cap.get(3)   # float
height = cap.get(4) # float
font = cv2.FONT_HERSHEY_SIMPLEX
sep = 5
cantidadDetecciones = 0

# classes num
#classesNum = [2,
#              1]


#fig, ax = plt.subplots()

#x_values = classesNum
#x_range = np.arange(len(x_values))
#ax.set_xticks(x_range)
#ax.set_xticklabels(x_values, rotation=45)
#plt.ion()

cam_params = {
    'brightness': cap.get(11), #32.0,
    'contrast': cap.get(12), #64.0,
    'saturation': cap.get(13), #0.0,
    'hue' : cap.get(14), #,
    'gain': cap.get(15), #,
    'exposure': cap.get(16), #,
    }

allowance = 0.9

# Print camera params
for i in cam_params:
    print('Param {} Value {}'.format(i, cam_params[i]))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    image = frame
    # ratio = 1 #frame.shape[0] / float(frame.shape[0])

    # convert to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    #thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_TOZERO_INV)[1]

    # find circles

    #circles=cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 0.1, 250)#
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=120,minRadius=20,maxRadius=350)

    # if there are circles
    if not circles is None:
        # for each circle...
        for i in circles[0,:]:

            # if circles redius is greater than 0
            if (i[2] > 0):
                (xPos, yPos, w, h) = ( int(i[0]-i[2]-sep),
                                 int(i[1]-i[2]-sep),
                                 int(i[2]*2) + sep,
                                 int(i[2]*2) + sep)

                if (xPos < 0):
                    xPos = 0

                if (yPos < 0):
                    yPos = 0

                #cv2.rectangle(image, (xPos,yPos), (xPos+w, yPos+h), (0, 0, 255), 2)
                #cv2.circle(image, (i[0], i[1]), i[2], (255,255,0), 2)

                # create mask
                mask = np.full((int(height), int(width)), 0, dtype=np.uint8)

                # draw white filled circle on the mask, where the coin is
                cv2.circle(mask, (i[0], i[1]), i[2], (255,255,255), -1)

                # show if debugging
                if debug:
                    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
                    cv2.imshow("mask", mask)

                # create white image
                clean1 = np.full((int(height), int(width), 3), (255,255,255), dtype=np.uint8)
                clean1[:] = (255, 255, 255)

                # get coin from image, on black background
                clean = cv2.bitwise_and(image, image, mask=mask)

                # invert mask (I am going to copy all white)
                mask = cv2.bitwise_not(mask)

                # clean1 will now get a black circle and white background
                clean1 = cv2.bitwise_and(clean1, clean1, mask=mask)

                if debug:
                    cv2.namedWindow('clean', cv2.WINDOW_NORMAL)
                    cv2.imshow("clean", clean)

                # add clean and clean1
                clean = cv2.add(clean, clean1)

                # get ROI - Region of Interest - Just the coin
                roi = clean[yPos:yPos+h, xPos:xPos+w]

                if debug:
                    cv2.namedWindow('roi')#, cv2.WINDOW_NORMAL)
                    cv2.imshow("roi", roi)

                # Resizing the image to our desired size and preprocessing will be done exactly as done during training
                imageFinal = cv2.resize(roi, (image_size, image_size),0,0, cv2.INTER_LINEAR)

                images = []
                images.append(imageFinal)
                images = np.array(images, dtype=np.uint8)
                images = images.astype('float32')
                images = np.multiply(images, 1.0/255.0)

                x_batch = images.reshape(1, image_size,image_size,num_channels)

                feed_dict_testing = {x: x_batch, y_true: y_test_images}

                result=sess.run(y_pred, feed_dict=feed_dict_testing)

                pred = np.argmax(result[0])

                if (result[0][pred] < allowance):
                    continue

                pred = switch(pred)

                #if plotting:
                    #tPos = 0
                    #y_values = np.arange(5)
                    #plt.pause(0.005)

                    #rects1 = ax.bar(x_range, y_values, 0.35, color='r')
                    #fig.canvas.draw()

                cv2.putText(frame, classesTxt[pred], (xPos + int(w) + 10, yPos + int(h/2)), font, 0.8, (0, 255, 0), 2, 1)
                print('\rCategoria : {0} --- P(f) = {1:>6.5%}'.format(classesTxt[pred], result[0][np.argmax(result[0])]), end = '')

                # uncomment to print images
                #cantidadDetecciones+=1
                #imgPath = dir_path + os.sep + 'training_data' + os.sep + 'out_' + str(pred) + '_' + str(cantidadDetecciones) +'_.png'
                #print(imgPath)
                #cv2.imwrite(imgPath, imageFinal)

    cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
    cv2.imshow("frame", frame)

    k = cv2.waitKey(1)
    if k==27:
        break
    elif k==ord('s'):
        sep += 5
        if (sep > 50):
            sep = 5
    elif k == ord('d'):
        if debug:
            debug = False
            printCamStats = False
        else:
            debug = True
            printCamStats = True
    #elif k == ord('r'):

        #plt.clf()
        #fig, ax = plt.subplots()
        #x_values = classesNum
        #x_range = np.arange(len(x_values))
        #ax.set_xticks(x_range)
        #ax.set_xticklabels(x_values, rotation=45)
        #plt.ion()

    elif k == ord('1'):
        cam_params['brightness'] += 1
        cap.set(11, cam_params['brightness'])
    elif k == ord('!'):
        cam_params['brightness'] -= 1
        cap.set(11, cam_params['brightness'])

    elif k == ord('2'):
        cam_params['contrast'] += 1
        cap.set(12, cam_params['contrast'])
    elif k == ord('"'):
        cam_params['contrast'] -= 1
        cap.set(12, cam_params['contrast'])

    elif k == ord('3'):
        cam_params['saturation'] += 1
        cap.set(13, cam_params['saturation'])
    elif k == ord('#'):
        cam_params['saturation'] -= 1
        cap.set(13, cam_params['saturation'])

    elif k == ord('p'):
        if plotting == True:
            plotting = False
        else:
            plotting = True

    elif k == ord('+'):
        if allowance > 0:
            allowance -= 0.001
            print('Allowance : {0:>6.5%}'.format(allowance))
    elif k == ord('-'):
        if allowance < 1:
            allowance += 0.001
            print('Allowance : {0:>6.5%}'.format(allowance))


    if printCamStats:
        print('  --- Cam Stats: {} {} {} {} {} {}'.format(
            cam_params['brightness'],
            cam_params['contrast'],
            cam_params['saturation'],
            cam_params['hue'],
            cam_params['gain'],
            cam_params['exposure']))

cap.release()
cv2.destroyAllWindows()
