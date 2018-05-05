#!/usr/bin/python
#-*- coding:utf-8 -*-
"""
Created on Wed Oct.16 2017

@author: Weisen pan

@function:
   face recognition
"""
__author__ = 'Weisen Pan'
import os
import sys
import numpy as np
import cv2

root_path = './cmti_libs/cnn_libs/' 
sys.path.insert(0, root_path + 'python')
os.environ['GLOG_minloglevel'] = '3'
import caffe
import matplotlib.pyplot as plt
import scipy 
import copy
import mysql.connector
from mysql.connector import errorcode
from datetime import datetime

from face_detection import FaceDetection

# move the DB configuration to a config file instead of hard code here
db_user = 'faces'
db_password = 'Faces123+-*/'
db_host = '127.0.0.1'
db_name = 'faces'
TMPFILE='TMP-IMAGE.JPEG'

# move the DB configuration to a config file instead of hard code here
db_user = 'faces'
db_password = 'Faces123+-*/'
db_host = '127.0.0.1'
db_name = 'faces'
TMPFILE='TMP-IMAGE.JPEG'

class FaceRecogniton(object):
    def __init__(self):
        super(FaceRecogniton, self).__init__()
        caffe.set_mode_cpu()
        # caffe.set_mode_gpu()
        # caffe.set_device(0)
        prototxt_a = r"./resource/.mod/face_recognition_A.prototxt"
        caffemodel_a = r"./resource/.mod/face_recognition_A.caffemodel"
        if not os.path.isfile(caffemodel_a):
          print ("caffemodel not found!")
        # self.th = 0.25
        self.net = caffe.Net(prototxt_a,caffemodel_a,caffe.TEST)

        prototxt_b = r"./resource/.mod/face_recognition_B.prototxt"
        caffemodel_b = r"./resource/.mod/face_recognition_B.caffemodel"

        self.face_det = FaceDetection()
        self.dict_feature = dict()
        self.dict_feature_by_faceset = dict()

    def extract_feature(self, img_color):
        img_gray = cv2.cvtColor( img_color, cv2.COLOR_RGB2GRAY )
        img_gray = cv2.resize(img_gray,(128,128),interpolation=cv2.INTER_CUBIC)
        img_blobinp = img_gray[np.newaxis, np.newaxis, :, :]/255.0
        self.net.blobs['data'].reshape(*img_blobinp.shape)
        self.net.blobs['data'].data[...] = img_blobinp
        self.net.blobs['data'].data.shape
        self.net.forward()
        feature = self.net.blobs['eltwise_fc1'].data
        return feature

    def load_target(self, folderpath):
        for parent,dirnames,filenames in os.walk(folderpath):
            # for dirname in  dirnames:
            #   # print "parent is:" + parent
            #   # print "dirname is" + dirname
            for filename in filenames:
                image_path = os.path.join(parent,filename)
                img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
                bbox_list = self.face_det.get_bounding_box_by_image(img_color)
                if bbox_list:
                    box = bbox_list[0]
                    crop_img = img_color[box[0][1]:box[1][1], box[0][0]:box[1][0]]
                    # cv2.imshow('image', crop_img)
                    # cv2.waitKey()
                    feat = self.extract_feature(crop_img)
                    copy_feat = copy.copy(feat[0])

                    self.dict_feature[image_path] = copy_feat
                    # print self.dict_feature
            # print self.dict_feature

    def load_target_image(self, image_path, file_name, overwriteFeatures = True):
        copy_feat = self.load_target_image_feature(file_name)
        if overwriteFeatures:
            self.dict_feature[image_path] = copy_feat
        return copy_feat.tostring()

    def load_target_image_with_facesetid(self, image_path, file_name, faceSetId):
        copy_feat = self.load_target_image_feature(file_name)
        if faceSetId in self.dict_feature_by_faceset:
            featDict = self.dict_feature_by_faceset[faceSetId]
        else:
            featDict = dict()
        featDict[image_path]=copy_feat
        self.dict_feature_by_faceset[faceSetId]=featDict
        return copy_feat.tostring()

    def load_target_image_feature(self, file_name):
        img_color = cv2.imread(file_name, cv2.IMREAD_COLOR)
        bbox_list = self.face_det.get_bounding_box_by_image(img_color)
        box = bbox_list[0]
        crop_img = img_color[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        feat = self.extract_feature(crop_img)
        return copy.copy(feat[0])

    def get_top_one_target(self, img_color):
        new_feat = self.extract_feature(img_color)
        dict_rank = {}

        for key in self.dict_feature:
            similar = 1 - scipy.spatial.distance.cosine(new_feat, self.dict_feature[key])   # similarity
            dict_rank[key] = similar
        # print dict_rank
        # sorted(dict_rank.items(),key = lambda x:x[1],reverse = False)
        top_one_path = max(dict_rank.items(), key=lambda x: x[1])[0]
        similarity = dict_rank[top_one_path]
        # print top_one_path
        # print similarity
        return top_one_path, similarity

    def get_top_one_target_with_facesetid(self, img_color, faceSetId):
        new_feat = self.extract_feature(img_color)
        dict_rank = {}
        top_one_path = None
        similarity = 0.0
        if faceSetId in self.dict_feature_by_faceset:
            featDict = self.dict_feature_by_faceset[faceSetId]
            for key in featDict:
                similar = 1 - scipy.spatial.distance.cosine(new_feat, featDict[key])   # similarity
                dict_rank[key] = similar

            top_one_path = max(dict_rank.items(), key=lambda x: x[1])[0]
            similarity = dict_rank[top_one_path]

        return top_one_path, similarity

    def load_target_image(self, image_path, file_name, overwriteFeatures = True):
        img_color = cv2.imread(file_name, cv2.IMREAD_COLOR)
        bbox_list = self.face_det.get_bounding_box_by_image(img_color)
        if bbox_list:
            box = bbox_list[0]
            crop_img = img_color[box[0][1]:box[1][1], box[0][0]:box[1][0]]
            feat = self.extract_feature(crop_img)
            copy_feat = copy.copy(feat[0])
            if overwriteFeatures:
                self.dict_feature[image_path] = copy_feat
            return copy_feat.tostring()

    def openDB(self):
        # DB init
        try:
            print str(datetime.now()).split('.')[0] + ' Open Database connection...'
            self.cnx = mysql.connector.connect(user=db_user, password=db_password, host=db_host, database=db_name)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Something is wrong with your user name:%s or password:%s", db_user, db_password)
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database %s does not exist", db_name)
            else:
                print(err)

            raise

    def load_target_faces_from_DB(self):
        try:
            self.openDB()
            queryStr = 'select id, name, photobinary, modifytime from tbl_faces where isDeleted != 1 order by modifytime desc'
            cursor = self.cnx.cursor()
            cursor.execute(queryStr)
            results = cursor.fetchall()
            cursor.close()
            if results:
                self.last_timestamp = results[0][3]
                print 'Saved last timestamp: ' + str(self.last_timestamp)
                for i in range(len(results)):
                    image_path = results[i][1] + '-' + str(results[i][0])
                    print image_path.encode('utf-8')
                    newFile = open(TMPFILE, "wb")
                    newFile.write(results[i][2])
                    self.load_target_image(image_path, TMPFILE)
            self.close()
        except mysql.connector.Error:
            print(mysql.connector.Error)
            raise

    def load_target_face_features_from_DB(self):
        try:
            self.openDB()
            queryStr = 'select id, name, featureBinary, modifytime from tbl_faces where isDeleted != 1 order by modifytime desc'
            cursor = self.cnx.cursor()
            cursor.execute(queryStr)
            results = cursor.fetchall()
            cursor.close()
            if results:
                self.last_timestamp = results[0][3]
                print 'Saved last timestamp: ' + str(self.last_timestamp)
                for i in range(len(results)):
                    image_path = results[i][1] + '-' + str(results[i][0])
                    print image_path.encode('utf-8')
                    # load the extracted features that are saved in DB to the dictionary
                    self.dict_feature[image_path] = np.frombuffer(results[i][2], dtype=np.float32)
            self.close()
        except mysql.connector.Error:
            print(mysql.connector.Error)
            raise

    def load_target_face_features_from_DB_groupby_facesetid(self):
        try:
            self.openDB()
            queryStr = 'select id, name, featureBinary, modifytime, facesetid from tbl_faces where isDeleted != 1 order by modifytime desc'
            cursor = self.cnx.cursor()
            cursor.execute(queryStr)
            results = cursor.fetchall()
            cursor.close()
            if results:
                self.last_timestamp = results[0][3]
                print 'Saved last timestamp: ' + str(self.last_timestamp)
                for i in range(len(results)):
                    if not results[i][2]:
                        continue
                    image_path = results[i][1] + '-' + str(results[i][0])
                    print image_path.encode('utf-8')
                    faceSetKey = str(results[i][4])
                    if faceSetKey in self.dict_feature_by_faceset:
                        featDict = self.dict_feature_by_faceset.get(faceSetKey)
                    else:
                        featDict = dict()

                    featDict[image_path]=np.frombuffer(results[i][2], dtype=np.float32)
                    self.dict_feature_by_faceset[faceSetKey]=featDict
            self.close()
        except mysql.connector.Error:
            print(mysql.connector.Error)
            raise

    def regenerate_features(self):
        try:
            self.openDB()
            queryStr = 'select id, name, photobinary from tbl_faces'
            cursor = self.cnx.cursor()
            cursor.execute(queryStr)
            results = cursor.fetchall()
            if results:
                for i in range(len(results)):
                    image_path = results[i][1] + '-' + str(results[i][0])
                    print image_path.encode('utf-8')
                    newFile = open(TMPFILE, "wb")
                    newFile.write(results[i][2])
                    features = self.load_target_image(image_path, TMPFILE)
                    update_faceinfo = ("update tbl_faces set modifytime = %s, featureBinary = %s where id = %s")

                    values = list()
                    values.append(datetime.now())
                    values.append(features)
                    values.append(results[i][0])
                    cursor.execute(update_faceinfo, tuple(values))

                    # Make sure data is committed to the database
                    self.cnx.commit()
            cursor.close()
            self.close()
        except mysql.connector.Error:
            print(mysql.connector.Error)
            raise

    def load_newer_target_faces_from_DB(self):
        try:
            self.openDB()
            cursor = self.cnx.cursor()

            if self.last_timestamp:
                queryStr = "select id, name, photobinary, modifytime from tbl_faces where modifytime > '" + str(self.last_timestamp) + "' order by modifytime desc"
            else:
                queryStr = "select id, name, photobinary, modifytime from tbl_faces where isDeleted != 1 order by modifytime desc"
            print queryStr
            cursor.execute(queryStr)
            results = cursor.fetchall()
            cursor.close()
            if results:
                self.last_timestamp = results[0][3]
                for i in range(len(results)):
                    # img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    # the blob data is the image binary
                    image_path = results[i][1] + '-' + str(results[i][0])
                    print image_path.encode('utf-8')
                    newFile = open(TMPFILE, "wb")
                    newFile.write(results[i][2])
                    self.load_target_image(image_path, TMPFILE)
            self.close()
        except mysql.connector.Error:
            print(mysql.connector.Error)
            raise

    def face_compare(self, img_a, img_b):
        a_feature = self.extract_feature(img_a)
        copy_a_feature = copy.copy(a_feature)
        b_feature = self.extract_feature(img_b)
        similarity = 1 - scipy.spatial.distance.cosine(copy_a_feature, b_feature)
        return similarity


    def face_recognition_1(self, img_a, img_b, gpu_id):
        #caffe.set_device(gpu_id)
        # img_a  = cv2.imread(img_a_path, cv2.IMREAD_COLOR) #Read image
        a_feature = self.extract_feature(img_a)
        copy_a_feature = copy.copy(a_feature)
        # print copy_a_feature
        # img_b = cv2.imread(img_b_path, cv2.IMREAD_COLOR) #Read image
        b_feature = self.extract_feature(img_b)
        # print b_feature
        similarity = 1 - scipy.spatial.distance.cosine(copy_a_feature, b_feature)
        # print similarity
        return similarity

    def face_recognition_n(self, img_path, target_folder, gpu_id):
        #caffe.set_device(gpu_id)
        self.load_target(target_folder)
        img_color  = cv2.imread(img_path, cv2.IMREAD_COLOR) #Read image
        new_feat = self.extract_feature(img_color)
        dict_rank = {}
        # print new_feat

        for key in self.dict_feature:
            test_feat = self.dict_feature[key]
            similar = 1 - scipy.spatial.distance.cosine(new_feat, self.dict_feature[key])   # similarity
            dict_rank[key] = similar
        top_one_path = max(dict_rank.items(), key=lambda x: x[1])[0]
        similarity = dict_rank[top_one_path]
        return similarity, top_one_path

    def close(self):
        """
        This is supposed to be the last call for the lifecycle
        Caller needs to call the close() method to release the DB connection
        :return: None
        """
        try:
            if self.cnx:
                self.cnx.close()
        except:
            raise

if __name__=='__main__':
    folderpath = './new-cmti-face'
    face_recon = FaceRecogniton()
    face_det = FaceDetection()
    face_recon.load_target(folderpath)
    image_path = 'weisen.png'
    img_cv = cv2.imread(image_path)
    bbox_list = face_det.get_bounding_box_by_image(img_cv)
    box = bbox_list[0]
    crop_img = img_cv[box[0][1]:box[1][1], box[0][0]:box[1][0]]
    top_one_path, similarity = face_recon.get_top_one_target(crop_img)
    image = cv2.imread(top_one_path)  
    cv2.imshow('face recognition', image)
    cv2.waitKey()