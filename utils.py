# -*- coding: UTF-8 -*-
"""
@Project ：Classified 
@File ：utils.py
@Author ：AnthonyZ
@Date ：2022/8/24 18:42
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import logging


def get_frames(filename, n_frames=1):
    frames = []
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list = np.linspace(0, v_len-1, n_frames+1, dtype=np.int16)
    for fn in range(v_len):
        success, frame = v_cap.read()
        if success is False:
            continue
        if fn in frame_list:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    v_cap.release()
    return frames, v_len


def store_frames(frames, path2store):
    for index, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        path2img = os.path.join(path2store, "frame"+str(index)+".jpg")
        cv2.imwrite(path2img, frame)


def pre_load(movie_path, pic_path, n_frames):
    listOfCategories = os.listdir(movie_path)
    print(listOfCategories, len(listOfCategories))
    for cat in listOfCategories:
        if cat == '.DS_Store':
            continue
        print("category:", cat)
        path2aCat = os.path.join(movie_path, cat)
        listOfSubs = os.listdir(path2aCat)
        print("number of sub-folders:", len(listOfSubs))
        print("-" * 50)

    extension = '.avi'
    for index, (root, dirs, files) in enumerate(os.walk(movie_path, topdown=False)):
        load_tqdm = tqdm(files, desc="Pre_loading:{}".format(index))
        for names in load_tqdm:
            if extension not in names:
                continue
            divided_path = os.path.join(root, names)
            frames, vlen = get_frames(divided_path, n_frames=n_frames)
            store_path = divided_path.replace(movie_path, pic_path)
            store_path = store_path.replace('.avi', '')
            os.makedirs(store_path, exist_ok=True)
            store_frames(frames, store_path)


def get_vids(frames_path):
    listOfCats = os.listdir(frames_path)
    # listOfCats.remove('.DS_Store')
    ids = []
    labels = []
    for index in listOfCats:
        if index == '.DS_Store':
            continue
        path2catg = os.path.join(frames_path, index)
        listOfSubCats = os.listdir(path2catg)
        path2subCats = [os.path.join(path2catg, los) for los in listOfSubCats]
        ids.extend(path2subCats)
        labels.extend([index]*len(listOfSubCats))
    return ids, labels, listOfCats


def init_logger(opt):
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')

    # 使用FileHandler输出到文件
    num_list = []
    for filename in os.listdir('./log'):
        if 'log' not in filename:
            continue
        num = int(filename.split('.')[0][3:])
        num_list.append(num)
    num = max(num_list) + 1 if num_list != [] else 1
    filename = os.path.join(opt.logdir, 'log{}.txt'.format(num))
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # 添加两个Handler
    logger.addHandler(fh)
    return logger


def args_logger(opt: dict, logdir):
    with open(logdir, "w") as f:
        for k, v in opt.items():
            f.write(k + " : " + v + "\n")


