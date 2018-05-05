#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, re, time
import _init_paths
import requests, urlparse
import mysql.connector
from mysql.connector import errorcode
from MysqlUtils import MysqlUtils
from random import *
import cv2

db_user = 'faces'
db_password = 'Faces123+-*/'
db_host = '127.0.0.1'
db_name = 'faces'
debug_mode = True

ID1='123456'
ID2='9876543'
NAME1='ABC'
NAME2='XYZ'
cnx = None

def openConnection():
    try:
        print 'Open Database connection...'
        cnx = mysql.connector.connect(user=db_user, password=db_password, host=db_host, database=db_name)
        instance = MysqlUtils(debug_mode=debug_mode, conn=cnx)
        return instance
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
        sys.exit(1)

def testRegen():
    url='http://127.0.0.1:6789/FaceRecognition/api/v1.1/regen'
    token = 'cmti'
    r = requests.post(url, params = {'token':token})
    print r.url
    print r.text

def testDetect(myname):
    url='http://127.0.0.1:6789/FaceRecognition/api/v1.1/detect'
    files = {'file': open('./test_image/target_face/' + myname +'.jpeg', 'rb')}
    r = requests.post(url, files=files)
    print r.url
    print r.text

def testCompare(person_a_name, person_b_name):
    url='http://127.0.0.1:6789/FaceRecognition/api/v1.1/compare'
    files = {'file_a': open('./test_image/cropped_faces/' + person_a_name +'.jpeg', 'rb'), 'file_b': open('./test_image/cropped_faces/' + person_b_name +'.jpeg', 'rb')}
    r = requests.post(url, files=files)
    print r.url
    print r.text

def testSearch(myname, faceSetId):
    url='http://127.0.0.1:6789/FaceRecognition/api/v1.1/search'
    files = {'file': open('./test_image/target_face/' + myname +'.jpeg', 'rb')}
    data_info = {'faceSetId': faceSetId}  
    r = requests.post(url, files=files, data=data_info)
    print r.url
    print r.text

def testDetectWithFaceSetId(myname, faceSetId):
    url='http://127.0.0.1:6789/FaceRecognition/api/v1.1/detect'
    files = {'file': open('./test_image/target_face/' + myname +'.jpeg', 'rb')}
    r = requests.post(url, files=files, params = {'faceSetId':faceSetId})
    print r.url
    print r.text

def testEnroll(myname):
    url='http://127.0.0.1:6789/FaceRecognition/api/v1.1/enroll'
    myid = randint(1, 1000000)
    files = {'file': open('./test_image/target_face/' + myname +'.jpeg', 'rb')}
    r = requests.post(url, files=files, params = {'name':myname, 'id':myid})
    print r.url
    print r.text

def testEnrollWithFaceSetId(myname, faceSetId):
    url='http://127.0.0.1:6789/FaceRecognition/api/v1.1/enroll'
    myid = randint(1, 1000000)
    files = {'file': open('./test_image/target_face/' + myname +'.jpeg', 'rb')}
    r = requests.post(url, files=files, params = {'name':myname, 'id':myid, 'faceSetId':faceSetId})
    print r.url
    print r.text

def testDeleteByNameOnline(name):
    url='http://127.0.0.1:6789/FaceRecognition/api/v1.1/deletebyname'
    r = requests.post(url, params = {'name':NAME1})
    print r.url
    print r.text

def testInsert(name, id):
    instance = openConnection()
    fileId = instance.insertFace(name, id)
    instance.close()
    print fileId

def testDeleteByName(name):
    instance = openConnection()
    fileId = instance.insertFace(name, 'junk')
    print fileId
    number = instance.deleteByName(name)
    instance.close()
    print number

def testDeleteByID(name, id):
    instance = openConnection()
    fileId = instance.insertFace(name, id)
    print fileId
    number = instance.deleteByID(id)
    instance.close()
    print number

def main(argv):
    myname='chenguang_wei'
    # myname='black'
    faceSetId=0
    #testEnrollWithFaceSetId(myname, faceSetId)
    # testDetectWithFaceSetId(myname, faceSetId)
    # testDetectWithFaceSetId(myname, faceSetId+1)
    testDetect(myname)
    person_a_name = 'Weisen_Pan'
    person_b_name = 'Weisen_Pan'
    testCompare(person_a_name, person_b_name)

    testSearch(myname, faceSetId)
    #testRegen()
    # testDeleteByNameOnline(NAME1)
    # testInsert(ID1, NAME1)
    # testDeleteByName(ID1)
    # testDeleteByID(NAME2, ID2)

if __name__ == '__main__':
    main(sys.argv)
