# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import sys
import time
import glob
import os
import cv2
import tensorflow as tf

import numpy as np
import detect_face

import data as dataset

from nn import GCN as ConvNet
from learning_utils import draw_pixel

import re
i=0
#graph = tf.get_default_graph()
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

root_dir = os.path.join('face_frames/')    # FIXME
test_dir = os.path.join(root_dir, 'video_face')

""" 2. Set test hyperparameters """
hp_d = dict()

# FIXME: Test hyperparameters
hp_d['batch_size'] = 8
IM_SIZE = (128, 128)
NUM_CLASSES = 3
origin_frame=[]
face_boundingbox=[]

""" 3. Build graph, load weights, initialize a session """
# Initialize

def main():
    frame_num=0
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0
    origin_frame=[]
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    video_capture = cv2.VideoCapture('./MANYFACE.mp4')

    fps=video_capture.get(cv2.CAP_PROP_FPS)
    width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out=cv2.VideoWriter("cvtest.avi",fourcc,fps,(int(width),int(height)))

    #face_recognition = face.Recognition()
    nrof_images_total = 0
    nrof_successfully_aligned = 0
    start_time = time.time()


    #if args.debug:
        #print("Debug enabled")
        #face.debug = True

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if ret:
            origin_frame.append(frame)
            bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            #if (frame_count % frame_interval) == 0:
                #faces = face_recognition.identify(frame)
            nrof_faces = bounding_boxes.shape[0]
            if nrof_faces > 0:
                det = bounding_boxes[:, 0:4]
                det_arr = []
                img_size = np.asarray(frame.shape)[0:2]
                if nrof_faces > 1:
                    if True: #args.multiface
                        for i in range(nrof_faces):
                            det_arr.append(np.squeeze(det[i]))
                    else:
                        bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                        img_center = img_size / 2
                        offsets = np.vstack(
                            [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                        index = np.argmax(
                            bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                        det_arr.append(det[index, :])
                else:
                    det_arr.append(np.squeeze(det))

                for i, det in enumerate(det_arr):
                    det = np.squeeze(det)
                    bb = np.zeros(4, dtype=np.int32)
                    det[0] = int(det[0])
                    det[1] = int(det[1])
                    det[2] = int(det[2])
                    det[3] = int(det[3])
                    det=det.astype(int)
                    bb[0] = np.maximum(det[0] - 44 / 2, 0) #args.margin
                    bb[1] = np.maximum(det[1] - 44 / 2, 0)
                    bb[2] = np.minimum(det[2] + 44 / 2, img_size[1])
                    bb[3] = np.minimum(det[3] + 44 / 2, img_size[0])
                    print("bb")
                    print(bb[3] - bb[1], bb[2] - bb[0])
                    cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]
                    #scaled = misc.imresize(cropped, (218, 178), interp='bilinear')
                    nrof_successfully_aligned += 1

                    #filename_base, file_extension = os.path.splitext(output_filename)

                # Check our current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0
            # add_overlays(frame, faces, frame_rate,frame_num)

            cv2.rectangle(frame, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])),(255,0,0),3)
            #detection 끝
            cv2.imshow('Video', frame)
            out.write(frame)
            filename=str(frame_count)+".jpg"

            face_boundingbox.append(det)

            faceframe=frame[int(det[1]):int(det[3]),int(det[0]):int(det[2]),:]
            cv2.imwrite("./face_frames/video_face/images/"+filename,faceframe)
            # cv2.imwrite(os.path.join("./face_frames",filename),faceframe)
            # face=frame[int(det[0]):int(det[1]),int(det[2]):int(det[3])]

            frame_num += 1
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break
    # with open('./bb.txt',"w") as f:
    #     for bbox in face_boundingbox:
    #         f.write(bbox)

    # When everything is done, release the capture
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()
    #여기까지 이미지 프레임별 저장

    X_test, y_test = dataset.read_data(test_dir, IM_SIZE, no_label=True)
    test_set = dataset.DataSet(X_test, y_test)

    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    model = ConvNet([IM_SIZE[0], IM_SIZE[1], 3], NUM_CLASSES, **hp_d)
    saver = tf.train.Saver()

    sess = tf.Session(graph=graph, config=config)
    saver.restore(sess, './checkpoint/model.ckpt')    # restore learned weights
    test_y_pred = model.predict(sess, test_set, **hp_d)

    """ 4. Draw boxes on image """
    draw_dir = os.path.join(test_dir, 'draws') # FIXME
    if not os.path.isdir(draw_dir):
        os.mkdir(draw_dir)
    im_dir = os.path.join(test_dir, 'images') # FIXME
    im_paths = []
    im_paths.extend(glob.glob(os.path.join(im_dir, '*.jpg')))
    test_outputs = draw_pixel(test_y_pred)
    test_results = test_outputs + test_set.images
    # test_results = test_outputs

    out2=cv2.VideoWriter("segtovideo.avi",fourcc,fps,(int(width),int(height)))

    for img, im_path in zip(test_results, im_paths):
        name = im_path.split('\\')[-1]
        draw_path =os.path.join(draw_dir, name)
        cv2.imwrite(draw_path, img)

    # imgs=[os.path.join(draw_dir,i) for i in os.listdir(draw_dir) if re.search(".jpg$",i)]
    # for image in imgs:
    #     image=cv2.imread(image)
    #     out2.write(image)
    # out2.release()

    ###########################segmentation 영상 생성 완료#####################################



    for original,img,bbox in zip(origin_frame,test_outputs,face_boundingbox):
        img[np.where((img!=[0,0,0]).all(axis=2))]=[20,20,0]
        # cv2.imshow("testttttt",img)
        # print("g")
        # print(bbox[1])
        # print(bbox[3])
        # print(bbox[0])
        # print(bbox[2])
        dif_x=((128-(bbox[3]-bbox[1]))/2)
        dif_y=(128-(bbox[2]-bbox[0]))/2
        # print(dif_x)
        # print(dif_y)
        # print(original.shape)
        face=original[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        mask=img[int(dif_x):int(128 - dif_x), int(dif_y):int(128 - dif_y)]

        if dif_x>0 and dif_y>0:
            idx=(mask!=0)
            face[idx]=mask[idx]
            # print(dif_x,dif_y)
            # bitwiseor = cv2.bitwise_or(original[bbox[1] + int(dif_x):bbox[3] - int(128 - dif_x),bbox[0] + int(dif_y):bbox[2] - int(128 - dif_y)], img)
            print(img[int(dif_x):int(128 - dif_x), int(dif_y):int(128 - dif_y)].shape)
            print(original[bbox[1]:bbox[3],bbox[0]:bbox[2]].shape)
            # cvadd=cv2.add(img[int(dif_x):int(128 - dif_x), int(dif_y):int(128 - dif_y)] , original[bbox[1]:bbox[3],bbox[0]:bbox[2]])
            cv2.imshow("img",img[int(dif_x):int(128 - dif_x), int(dif_y):int(128 - dif_y)])
            cv2.imshow("original",original[bbox[1]:bbox[3],bbox[0]:bbox[2]])
            cv2.imshow("output",original)
            out2.write(original)
            # cv2.imshow("bitwise",bitwiseor)
        else :
            pass
            #이외의 예외처리 진행해야함
            # print(dif_x, dif_y)
            # bitwiseor = cv2.bitwise_or(original[bbox[1] + int(dif_x):bbox[3] - int(128 - dif_x),bbox[0] + int(dif_y):bbox[2] - int(128 - dif_y)], img)
            # cv2.imshow("bitwise", bitwiseor)
            # cv2.imshow("output", original[bbox[1] + int(dif_x):bbox[3] - int(128 - dif_x),bbox[0] + int(dif_y):bbox[2] - int(128 - dif_y)] + img)
            # cv2.imshow("test",original[bbox[1]:bbox[3],bbox[0]:bbox[2]])
        # cv2.imshow("test", original[bbox[1]:bbox[3],bbox[0]:bbox[2]])
        # bitwiseor=cv2.bitwise_or(original[bbox[1]:bbox[3],bbox[0]:bbox[2]],img)
        # cv2.imshow(bitwiseor)
        time.sleep(1/fps)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out2.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
