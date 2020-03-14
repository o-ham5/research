# -*- coding: utf-8 -*-

import glob
import cv2
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import sqrt
from time import time
import pickle
import os
import argparse
from itertools import combinations, product
from collections import defaultdict

from calib_models import Single_Camera, Stereo_Camera
from evaluator import SingleCamera_Evaluator, StereoCamera_Evaluator

from math import sqrt, atan



# print(cv2.__version__)

def main():
    ################################################
    #------------------ 手順 -----------------------#
    ################################################

    # 0. パラメータ設定

    ##### 各カメラにおいて #####
    # 1. チェスボードを撮影した画像を読み込む
    # 2. チェスボードの交点検出
    # 3. 検出した画像座標上の点に対応する3次元上の点を作成する
    # 4. キャリブレーション
    # 5. World -> Image (1つのカメラで)
    # 6. Image -> World (1つのカメラで)
    # (7. 内部パラメータ固定でカメラ移動)

    ##### ２カメラにおいて #####
    # 1. チェスボードを撮影した画像を読み込む
    # 2. チェスボードの交点検出
    # 3. 検出した画像座標上の点に対応する3次元上の点を作成する
    # 4. キャリブレーション
    # 5. 別カメラへの変換
    # 6. Image -> World (2つのカメラで)

    # -------------------------------------------------- #



    ##### 0. パラメータ設定 #####

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--root_dir', required=True, help='root directory path')
    parser.add_argument('-K', '--K_dir', default=None, help='params directory path')
    parser.add_argument('--fish', action='store_true', help='')
    parser.add_argument('-c', '--cols', default=4, help='')
    parser.add_argument('-r', '--rows', default=3, help='')
    parser.add_argument('-s', '--size', default=200, help='')
    parser.add_argument('--draw1', nargs='*', default=None, help='[example] cam5 5 3')
    parser.add_argument('--draw2', nargs=3, default=None, help='[example] cam5 cam7 5')
    parser.add_argument('-w', '--write', action='store_true', help='')

    args = parser.parse_args()

    root_dirpath, K_dirpath, fisheye, rows, cols, chess_size, arg_draw1, arg_draw2, write = args.root_dir, args.K_dir, args.fish, int(args.cols), int(args.rows), float(args.size), args.draw1, args.draw2, args.write


    # 結果の出力先
    result = root_dirpath + '/result/'
    output = result + 'output/'
    if not os.path.exists(result):
        os.mkdir(result)
    if not os.path.exists(output):
        os.mkdir(output)


    # 画像パス設定

    Cam_dirpaths = {path.split('/')[-1]:path for path in glob.glob(root_dirpath + '/images/*')}
    

    paths_list_1cam = {}
    paths_list_2cam = {}

    # 各1カメラの画像データパス保存
    for cam_name, path in Cam_dirpaths.items():
        path_images = glob.glob(path + '/*.png')
        path_images = sorted(path_images, key=lambda x:int(x.split('/')[-1].split('.')[0]))
        paths_list_1cam[cam_name] = path_images



    # 隣接カメラ情報ファイルの取得
    nei_path = f'{root_dirpath}/neighbor.csv'
    nei_df = pd.read_csv(nei_path)
    nei_df['cam'] = nei_df['cam'].astype(str)
    nei_df = nei_df.set_index('cam')

    # ディレクトリ内のカメラ名とneighbor.csv内のカメラ名が異なった場合にはエラー
    if not set(nei_df.index) == set(Cam_dirpaths.keys()):
        print(f'Cam list in dirs = {set(Cam_dirpaths.keys())}')
        print(f'Cam list in neighbor file = {set(nei_df.index)}')
        print('Neighbor file error.')
        exit()


    # 各2カメラの画像データパス保存
    for (i, cam1), (j, cam2) in product(enumerate(nei_df.index), enumerate(nei_df.index)):
        if i >= j or nei_df[cam1][cam2] != 1:
            continue
        pathi_j = paths_list_1cam[cam1]
        pathj_i = paths_list_1cam[cam2]
        pathi_j = sorted(pathi_j, key=lambda x:int(x.split('/')[-1].split('.')[0]))
        pathj_i = sorted(pathj_i, key=lambda x:int(x.split('/')[-1].split('.')[0]))

        paths_list_2cam[(cam1, cam2)] = (pathi_j, pathj_i)



    draw1_flag = defaultdict(lambda:defaultdict(dict))
    draw2_flag = defaultdict(dict)

    for cam_name in paths_list_1cam.keys():
        draw1_flag[cam_name][5][1] = False
        draw1_flag[cam_name][5][2] = False
        draw1_flag[cam_name][5][3] = False
        draw1_flag[cam_name][6] = False

    for cam1_name, cam2_name in paths_list_2cam.keys():
        draw2_flag[(sorted([cam1_name, cam2_name])[0], sorted([cam1_name, cam2_name])[1])][5] = False
        draw2_flag[(sorted([cam1_name, cam2_name])[0], sorted([cam1_name, cam2_name])[1])][6] = False

    if arg_draw1:
        if not arg_draw1[0] in set(paths_list_1cam.keys()):
            print('invalid camera name in draw1')
            exit()
        if len(arg_draw1) == 2:
            cam_name, i = arg_draw1
            draw1_flag[cam_name][int(i)] = True
        elif len(arg_draw1) == 3:
            cam_name, i, j = arg_draw1
            draw1_flag[cam_name][int(i)][int(j)] = True

    if arg_draw2:
        if not arg_draw2[0] in set(paths_list_1cam.keys()) or not arg_draw2[1] in set(paths_list_1cam.keys()):
            print('invalid camera name in draw2')
            exit()
        cam1_name, cam2_name, i = arg_draw2
        draw2_flag[(sorted([cam1_name, cam2_name])[0], sorted([cam1_name, cam2_name])[1])][int(i)] = True


    ################################################
    #---------- 各カメラでキャリブレーション -----------#
    ################################################


    print('-'*30)
    print('single camera calibration')
    print('-'*30, '\n')

    t_1_1 = []
    t_1_2 = []
    t_1_3 = []
    t_1_4 = []
    t_1_5_1 = []
    t_1_5_2 = []
    t_1_5_3 = []
    t_1_6 = []

    t_2_1 = []
    t_2_2 = []
    t_2_3 = []
    t_2_4 = []
    t_2_5 = []
    t_2_6 = []

    # ２カメラにも使う
    def read_Cam(path, fisheye):
        f = open(path, 'rb')
        cam = pickle.load(f)
        cam.fisheye = fisheye
        
        return cam

    save_flag = False
    if os.path.exists(result + 'models'):
        Cams = {cam_name: read_Cam(f'{result}models/Cam{cam_name}', fisheye) for cam_name in paths_list_1cam.keys()}
        
        save_flag = True
    
    else:

        Cams = {cam_name: Single_Camera(paths, cols, rows, chess_size, fisheye=fisheye) for cam_name, paths in paths_list_1cam.items()}
        
        os.mkdir(result + 'models')

        for k, (cam_name, Cam) in enumerate(Cams.items()):

            print(f'- cam{cam_name} -')

            ##### 1. チェスボードを撮影した画像を読み込む #####

            print('--- 1. 画像読み込み ---\n')

            ts = time()

            Cam.read_file()
            print(f'画像数 : {len(Cam.imgs_all)}\n')

            te = time()
            t_1_1.append(te-ts)

            ##### 2. チェスボードの交点検出 #####

            print('--- 2. チェッカーボードの交点検出 ---\n')

            ts = time()

            Cam.find_chessCorner()
            print(f'使用する画像数 : {len(Cam.imgs)}\n')
            print()

            te = time()
            t_1_2.append(te-ts)


            ##### 3. 検出した画像座標上の点に対応する3次元上の点を作成する #####

            print('--- 3. 対応する世界座標点生成 ---\n')

            ts = time()

            Cam.make_worldPoint()

            te = time()
            t_1_3.append(te-ts)
            
            f = open(f'{result}models/Cam{cam_name}', 'wb')
            pickle.dump(Cam, f)
            

    for k, (cam_name, Cam) in enumerate(Cams.items()):

        print(f'- cam{cam_name} -')

        #####  4. キャリブレーション  #####

        print('--- 4. キャリブレーション ---\n')


        if K_dirpath:
            f = open(f"{K_dirpath}/Cam{cam_name}", 'rb')
            pre_Cam = pickle.load(f)
            Cam.reprojection_error = pre_Cam.reprojection_error
            Cam.camera_matrix = pre_Cam.camera_matrix
            Cam.distortion = pre_Cam.distortion

            rvecs, tvecs = Cam.estimate_extrinsicParams(Cam.camera_matrix, Cam.distortion)
            Cam.rvecs = rvecs
            Cam.tvecs = tvecs

        else:
            ts = time()

            Cam.calibration()

            te = time()
            t_1_4.append(te-ts)

        f = open(f'{result}models/Cam{cam_name}', 'wb')
        pickle.dump(Cam, f)
            
        print(f'再投影誤差 =\n{Cam.reprojection_error}\n')
        print(f'カメラ行列 =\n{Cam.camera_matrix}\n')
        print(f'レンズ歪み =\n{Cam.distortion}\n')




    # 適当なカメラのパラメータを使用
    tmp = int(len(Cams)/2)
    sample_cam = list(Cams.values())[tmp]
    
    # 評価
    for k, (cam_name, Cam) in enumerate(Cams.items()):

        print(f'- cam{cam_name} -')

        # 内部・歪みパラメータ固定で外部パラメータを推定

        rvecs_fixK, tvecs_fixK = Cam.estimate_extrinsicParams(sample_cam.camera_matrix, sample_cam.distortion)

    
        #####  5. World -> Image (1つのカメラで)  #####

        evaluator = SingleCamera_Evaluator(Cam, fisheye=fisheye)

        print('--- 5. World -> Image (1カメラ) ---\n')

        ts = time()

        # １カメラ内の変換( World -> Image, 正解パラメータ)
        title = f'正解パラメータでの誤差計算(cam{cam_name})'
        print(title + '\n')

        evaluator.evaluate_World2Image(draw_flag=draw1_flag[cam_name][5][1], title=title, output=output, write=write)

        te = time()
        t_1_5_1.append(te-ts)

        ts = time()

        # １カメラ内の変換( World -> Image, 別カメラパラメータ)
        title = f'別カメラパラメータでの誤差計算(cam{cam_name})'
        print(title + '\n')

        evaluator.evaluate_World2Image(camera_matrix_2=sample_cam.camera_matrix, distortion_2=sample_cam.distortion, draw_flag=draw1_flag[cam_name][5][2], title=title, output=output, write=write)

        te = time()
        t_1_5_2.append(te-ts)

        ts = time()

        # １カメラ内の変換( World -> Image, 別カメラ内部・歪みパラメータ, 外部パラメータも推定したもの)
        title = f'別カメラパラメータでの誤差計算(外部は改めて推定)(cam{cam_name})'
        print(title + '\n')

        evaluator.evaluate_World2Image(rvecs_2=rvecs_fixK, tvecs_2=tvecs_fixK, camera_matrix_2=sample_cam.camera_matrix, distortion_2=sample_cam.distortion, draw_flag=draw1_flag[cam_name][5][3], title=title, output=output, write=write)
        print()

        te = time()
        t_1_5_3.append(te-ts)

        #####  6. Image -> World (1つのカメラで) #####

        print('--- 6. Image -> World (1カメラ) ---\n')

        ts = time()

        title = f'１カメラ内での変換( Image -> World )による誤差計算(cam{cam_name})'
        print(f'---\n{title}\n')

        evaluator.evaluate_Image2World(draw_flag=draw1_flag[cam_name][6], title=title, output=output, write=write)

        te = time()
        t_1_6.append(te-ts)

        print()



    ################################################
    #----------- ここからステレオマッチング ------------#
    ################################################

    print('-'*30)
    print('stereo camera calibration')
    print('-'*30, '\n')


    if save_flag:
        s_Cams = {(cam1_name, cam2_name): read_Cam(f'{result}models/Cam{cam1_name}_{cam2_name}', fisheye) for cam1_name, cam2_name in paths_list_2cam.keys()}

        if K_dirpath:
            for (cam1_name, cam2_name), s_Cam in s_Cams.items():
                s_Cam.camera_matrix1 = Cams[cam1_name].camera_matrix
                s_Cam.distortion1 = Cams[cam1_name].distortion
                s_Cam.camera_matrix2 = Cams[cam2_name].camera_matrix
                s_Cam.distortion2 = Cams[cam2_name].distortion
    
    else:

        s_Cams = {(cam1_name, cam2_name): Stereo_Camera(cam1_paths, cam2_paths, cols, cols, rows, rows, chess_size, chess_size, Cams[cam1_name].camera_matrix, Cams[cam2_name].camera_matrix, Cams[cam1_name].distortion, Cams[cam2_name].distortion, fisheye=fisheye) for (cam1_name, cam2_name), (cam1_paths, cam2_paths) in paths_list_2cam.items()}
        

        for k, ((cam1_name, cam2_name), s_Cam) in enumerate(s_Cams.items()):

            print(f'- cam{cam1_name}, cam{cam2_name} -')

            ##### 1. チェスボードを撮影した画像を読み込む #####

            print('--- 1. 画像読み込み ---\n')

            ts = time()

            s_Cam.read_file()
            
            print(f'画像数 :', len(s_Cam.imgs1_all), '\n')

            te = time()
            t_2_1.append(te-ts)

            ##### 2. チェスボードの交点検出 #####

            print('--- 2. チェッカーボードの交点検出 ---\n')

            ts = time()

            s_Cam.find_chessCorner()
            print('使用する画像数 :', len(s_Cam.imgs1), '\n')
            
            te = time()
            t_2_2.append(te-ts)


            ##### 3. 検出した画像座標上の点に対応する3次元上の点を作成する #####
            # これは上で作成したもの(world_points)と同様なので使い回し

            print('--- 3. 対応する世界座標点生成 ---\n')

            ts = time()

            s_Cam.make_worldPoint()

            te = time()
            t_2_3.append(te-ts)


            f = open(f'{result}models/Cam{cam1_name}_{cam2_name}', 'wb')
            pickle.dump(s_Cam, f)


    for k, ((cam1_name, cam2_name), s_Cam) in enumerate(s_Cams.items()):

        print(f'- cam{cam1_name}, cam{cam2_name} -')

        #####  4. キャリブレーション  #####

        print('--- 4. ステレオキャリブレーション ---\n')

        ts = time()

        # ステレオキャリブレーション

        s_Cam.stereo_calibration()
        print(f"stereo reprojection error = \n{s_Cam.projection_error}\n")
        print(f"R = \n{s_Cam.R}\n")
        print(f"T = \n{s_Cam.T}\n")

        f = open(f'{result}models/Cam{cam1_name}_{cam2_name}', 'wb')
        pickle.dump(s_Cam, f)

        te = time()
        t_2_4.append(te-ts)


        s_Cam.estimate_extrinsicParams()

        s_evaluator = StereoCamera_Evaluator(s_Cam)

        if s_Cam.fisheye:
            title = f"外部パラメータの固定({cam1_name}, {cam2_name})"
            print(f'---\n{title}\n')
            
            s_evaluator.fix_Rt(draw_flag=False, title=title, output=output, write=write)


        #####  5. 別カメラへの変換  #####

        print('--- 5. World -> Image (2カメラ) ---\n')

        # 基本行列によるカメラ座標変換
        ts = time()
            
        title = f'基本行列による別カメラ変換( world -> Image )での誤差計算({cam2_name} -> {cam1_name})'
        print(f'---\n{title}\n')

        s_evaluator.evaluate_World2Image_by_RT(draw_flag=draw2_flag[(sorted([cam1_name, cam2_name])[0], sorted([cam1_name, cam2_name])[1])][5], title=title, output=output, write=write)

        te = time()
        t_2_5.append(te-ts)


        #####  6. Image -> World (2つのカメラで) #####

        print('--- 6. Image -> World (2カメラ) ---\n')

        ts = time()

        title = f'別カメラへの変換( Image -> World )による誤差計算({cam2_name} -> {cam1_name})'
        print(f'---\n{title}\n')

        s_evaluator.evaluate_Image2World(draw_flag=draw2_flag[(sorted([cam1_name, cam2_name])[0], sorted([cam1_name, cam2_name])[1])][6], title=title, output=output, write=write)

        te = time()
        t_2_6.append(te-ts)


    # 計測時間

    if not save_flag:
        print('---\nプロファイル\n')

        print('--- 各カメラでキャリブレーション ---\n')
        for k, cam_name in enumerate(Cams.keys()):
            print(f'- cam{cam_name} -')
            print('画像読み込み :', t_1_1[k], '[s]')
            print('交点検出 :', t_1_2[k], '[s]')
            print('世界座標点生成 :', t_1_3[k], '[s]')
            print('キャリブレーション :', t_1_4[k], '[s]')
            print('1カメラ内変換1 (world -> image) :', t_1_5_1[k], '[s]')
            print('1カメラ内変換2 (world -> image) :', t_1_5_2[k], '[s]')
            print('1カメラ内変換3 (world -> image) :', t_1_5_3[k], '[s]')
            print('1カメラ内変換 (image -> world) :', t_1_6[k], '[s]')
            print()

        print()

        print('--- ステレオキャリブレーション ---\n')
        for k, (cam1_name, cam2_name) in enumerate(s_Cams):
            print(f'- cam{cam1_name}, cam{cam2_name} -')
            print('画像読み込み :', t_2_1[k], '[s]')
            print('交点検出 :', t_2_2[k], '[s]')
            print('世界座標点生成 :', t_2_3[k], '[s]')
            print('ステレオキャリブレーション :', t_2_4[k], '[s]')
            print('別カメラ変換 (world -> image) :', t_2_5[k], '[s]')
            print('別カメラ変換 (image -> world) :', t_2_6[k], '[s]')
            print()



def World2Camera(rvec, tvec, p):
    '''
    input:
    rvec = [[x], [y], [z]] : 回転ベクトル
    tvec = [[x], [y], [z]] : 並進ベクトル
    p = [x, y, z] : 対象点

    output:
    trans_p = [x, y, z] : 移動後の点
    '''

    rvec = rvec.reshape(1, -1)[0]
    tvec = tvec.reshape(1, -1)[0]
    rvec_ = rvec / np.linalg.norm(rvec, ord=2)
    theta = np.linalg.norm(rvec, ord=2)
    trans_p = p*np.cos(theta) + rvec_*np.dot(rvec_, p)*(1-np.cos(theta)) + np.cross(rvec_, p)*np.sin(theta)
    trans_p = trans_p + tvec

    return trans_p

def calc_fisheye_distortion(distortion, p):
    k1, k2, k3, k4 = distortion[0][0], distortion[1][0], distortion[2][0], distortion[3][0]
    x, y, z = p
    a = x / z
    b = y / z
    r = sqrt((a*a) + (b*b))
    shita = atan(r)

    shita_d = shita*(1 + k1*(shita**2) + k2*(shita**4) + k3*(shita**6) + k4*(shita**8))

    x_ = (shita_d / r)*a
    y_ = (shita_d / r)*b

    return (x_, y_)

def Camera2Image(cameraMatrix, p):
    '''
    内部パラメータを用いて カメラ座標 -> 画像座標 の変換を計算する関数
    '''
    f_x, f_y, c_x, c_y = cameraMatrix[0][0], cameraMatrix[1][1], cameraMatrix[0][2], cameraMatrix[1][2]
    x, y = p
    x = f_x*x + c_x
    y = f_y*y + c_y

    return (x, y)


def Camera2otherCamera(rmat, tvec, p):
    '''
    input:
    rmat = [[x, y, z], ...] : 回転行列
    tvec = [[x], [y], [z]] : 並進ベクトル
    p = [x, y, z] : 対象点
    
    output:
    trans_p = [x, y, z] : 移動後の点
    '''

    tvec = tvec.reshape(1, -1)[0]
    trans_p = np.dot(rmat, p) + tvec

    return trans_p




if __name__ == '__main__':
    main()