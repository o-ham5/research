import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, atan


class SingleCamera_Evaluator():
    '''
    １カメラのキャリブレーションに関するクラス
    '''
    def __init__(self, Cam, fisheye=False):
        self.Cam = Cam
        self.fisheye = fisheye

    def evaluate_World2Image(self, rvecs_2=None, tvecs_2=None, camera_matrix_2=None, distortion_2=None, draw_flag=False, title='title', output='./', write=False):
        total_error = 0
        error_list = []
        for i in range(len(self.Cam.object_points)):
            _img = self.Cam.imgs[i]

            mini_total_error = 0

            for pos, point in zip(self.Cam.image_points[i], self.Cam.object_points[i]):

                # 正解点のプロット
                if draw_flag:
                    _img = cv2.drawMarker(_img, tuple(pos[0]), color=(0, 255, 0), markerType=cv2.MARKER_SQUARE, thickness=1, line_type=cv2.LINE_4)
            
                # 予測点の変換
                if rvecs_2 is not None and tvecs_2 is not None:
                    point = World2Camera(rvecs_2[i], tvecs_2[i], point)
                else:
                    point = World2Camera(self.Cam.rvecs[i], self.Cam.tvecs[i], point)     


                if distortion_2 is not None:
                    if self.fisheye:
                        point = calc_fisheye_distortion(distortion_2, point)
                    else:
                        point = calc_distortion(distortion_2, point)
                else:
                    if self.fisheye:
                        point = calc_fisheye_distortion(self.Cam.distortion, point)
                    else:
                        point = calc_distortion(self.Cam.distortion, point)
                if camera_matrix_2 is not None:
                    point = Camera2Image(camera_matrix_2, point)
                else:
                    point = Camera2Image(self.Cam.camera_matrix, point)

                # 予測点のプロット
                if draw_flag:
                    _point = list(map(int, point))
                    # print(_point)
                    # try:
                    _img = cv2.drawMarker(_img, tuple(_point), color=(0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, thickness=1, line_type=cv2.LINE_4)
                    # except:
                    #     print("error")
            

                # 誤差計算
                mini_total_error += np.linalg.norm(pos[0] - point)

            # 誤差計算
            mini_total_error = mini_total_error / len(self.Cam.object_points[i])

            if draw_flag:
                print(f'画像{i+1}の誤差合計 = {mini_total_error}\n')
                plt.imshow(_img)
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                plt.axis('off')
                plt.show()

            total_error += mini_total_error
            error_list.append(mini_total_error)

        total_error = total_error / len(self.Cam.object_points)
        print('全画像の誤差平均 =', total_error, '\n')
        
        if write:
            draw_error(error_list, title, output, total_error)

    def evaluate_Image2World(self, draw_flag=False, title='title', output='./', write=False):
        total_error = 0
        error_list = []

        for k in range(len(self.Cam.object_points)):

            if draw_flag:
                _img = self.Cam.imgs[k]

            if self.fisheye:
                undistorted_point = cv2.fisheye.undistortPoints(self.Cam.image_points[k], self.Cam.camera_matrix, self.Cam.distortion, P=self.Cam.camera_matrix)
            else:
                undistorted_point = cv2.undistortPoints(self.Cam.image_points[k], self.Cam.camera_matrix, self.Cam.distortion, P=self.Cam.camera_matrix)

            mini_total_error = 0
            for i in range(len(self.Cam.object_points[k])):

                # 世界座標の正解点
                w_p = self.Cam.object_points[k][i]


                # 画像座標の正解点(対応点)を世界座標に変換
                p = undistorted_point[i][0]
                p = np.r_[p, [1]]
                p = Image2Camera(self.Cam.camera_matrix, p)
                p = homogeneous2point3d(p, self.Cam.rvecs[k], self.Cam.tvecs[k])
                p = Camera2World(self.Cam.rvecs[k], self.Cam.tvecs[k], p)

                # 誤差計算
                mini_total_error += np.linalg.norm(w_p - p)

                if draw_flag:
                    if self.fisheye:
                        draw_w_p, _ = cv2.fisheye.projectPoints(w_p[np.newaxis, np.newaxis, :], self.Cam.rvecs[k], self.Cam.tvecs[k], self.Cam.camera_matrix, self.Cam.distortion)
                        

                        draw_p, _ = cv2.fisheye.projectPoints(p[np.newaxis, np.newaxis, :], self.Cam.rvecs[k], self.Cam.tvecs[k], self.Cam.camera_matrix, self.Cam.distortion)
                        
                    else:
                        draw_w_p, _ = cv2.projectPoints(w_p[np.newaxis, :], self.Cam.rvecs[k], self.Cam.tvecs[k], self.Cam.camera_matrix, self.Cam.distortion)

                        draw_p, _ = cv2.projectPoints(p[np.newaxis, :], self.Cam.rvecs[k], self.Cam.tvecs[k], self.Cam.camera_matrix, self.Cam.distortion)
                    
                    draw_w_p = list(map(int, draw_w_p[0][0]))
                    _img = cv2.drawMarker(_img, tuple(draw_w_p), color=(255, 0, 0), markerType=cv2.MARKER_SQUARE, thickness=1, line_type=cv2.LINE_4)
                    
                    draw_p = list(map(int, draw_p[0][0]))
                    _img = cv2.drawMarker(_img, tuple(draw_p), color=(0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, thickness=1, line_type=cv2.LINE_4)


            # 誤差計算
            mini_total_error = mini_total_error / len(self.Cam.object_points[k])
            
            total_error += mini_total_error
            error_list.append(mini_total_error)

            if draw_flag:
                print(f'画像{k+1}の誤差合計 = {mini_total_error}\n')
                plt.imshow(_img)
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                plt.axis('off')
                plt.show()

        total_error = total_error / len(self.Cam.object_points)
        print('全画像の誤差平均 =', total_error, '\n')

        if write:
            draw_error(error_list, title, output, total_error)

class StereoCamera_Evaluator():
    '''
    ２カメラのキャリブレーションに関するクラス
    '''
    def __init__(self, Cam1_2):
        self.Cam1_2 = Cam1_2
        self.Rt1_num = None
        self.Rt2_num = None

    def fix_Rt(self, draw_flag=False, title='title', output='./', write=False):
        '''
        
        '''

        total_error = 0
        error_list = []

        for i in range(len(self.Cam1_2.object_points1)):
            _img = self.Cam1_2.imgs1[i]

            mini_total_error = 0
            for pos, point in zip(self.Cam1_2.image_points1[i], self.Cam1_2.object_points1[i]):

                # 正解点のプロット
                if draw_flag:
                    _img = cv2.drawMarker(_img, tuple(pos[0]), color=(0, 255, 0), markerType=cv2.MARKER_SQUARE, thickness=1, line_type=cv2.LINE_4)

                point = World2Camera(self.Cam1_2.rvecs1[i], self.Cam1_2.tvecs1[i], point)

                point = calc_fisheye_distortion(self.Cam1_2.distortion1, point)
                
                point = Camera2Image(self.Cam1_2.camera_matrix1, point)

                # 予測点のプロット
                if draw_flag:
                    _point = list(map(int, point))
                    _img = cv2.drawMarker(_img, tuple(_point), color=(0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, thickness=1, line_type=cv2.LINE_4)
            

                # 誤差計算
                mini_total_error += np.linalg.norm(pos[0] - point)

            # 誤差計算
            mini_total_error = mini_total_error / len(self.Cam1_2.object_points1[i])


            if draw_flag:
                print(f'画像{i+1}の誤差合計 = {mini_total_error}\n')
                plt.imshow(_img)
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                plt.axis('off')
                plt.show()

            total_error += mini_total_error
            error_list.append(mini_total_error)

        min_error = min(error_list)
        min_index = error_list.index(min_error)
        self.Rt1_num = min_index
        print(f"min error(cam1) = {min_error}")

        total_error = total_error / len(self.Cam1_2.object_points1)
        print('全画像の誤差平均 =', total_error, '\n')

        if write:
            draw_error(error_list, title+"_1", output, total_error)

        # ---------------------

        total_error = 0
        error_list = []

        for i in range(len(self.Cam1_2.object_points2)):
            _img = self.Cam1_2.imgs2[i]

            mini_total_error = 0
            for pos, point in zip(self.Cam1_2.image_points2[i], self.Cam1_2.object_points2[i]):

                # 正解点のプロット
                if draw_flag:
                    _img = cv2.drawMarker(_img, tuple(pos[0]), color=(0, 255, 0), markerType=cv2.MARKER_SQUARE, thickness=1, line_type=cv2.LINE_4)

                point = World2Camera(self.Cam1_2.rvecs2[i], self.Cam1_2.tvecs2[i], point)

                point = calc_fisheye_distortion(self.Cam1_2.distortion2, point)
                
                point = Camera2Image(self.Cam1_2.camera_matrix2, point)

                # 予測点のプロット
                if draw_flag:
                    _point = list(map(int, point))
                    _img = cv2.drawMarker(_img, tuple(_point), color=(0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, thickness=1, line_type=cv2.LINE_4)
            

                # 誤差計算
                mini_total_error += np.linalg.norm(pos[0] - point)

            # 誤差計算
            mini_total_error = mini_total_error / len(self.Cam1_2.object_points2[i])


            if draw_flag:
                print(f'画像{i+1}の誤差合計 = {mini_total_error}\n')
                plt.imshow(_img)
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                plt.axis('off')
                plt.show()

            total_error += mini_total_error
            error_list.append(mini_total_error)

        min_error = min(error_list)
        min_index = error_list.index(min_error)
        self.Rt2_num = min_index
        print(f"min error(cam2) = {min_error}")

        total_error = total_error / len(self.Cam1_2.object_points2)
        print('全画像の誤差平均 =', total_error, '\n')

        if write:
            draw_error(error_list, title+"_2", output, total_error)

    
    def evaluate_World2Image_by_RT(self, draw_flag=False, title='title', output='./', write=False):
        '''
        基本行列を用いてカメラ2の点をカメラ1の点に変換し, 描写する関数
        <注意> 描写に関しては，描写関数の都合上画像点をintで丸め込んでいるのでズレが実際より大きい
        '''

        total_error = 0
        error_list = []

        for i in range(len(self.Cam1_2.object_points1)):

            if self.Rt2_num is not None:
                Rt_idx = self.Rt2_num
            else:
                Rt_idx = i

            _img = self.Cam1_2.imgs1[i]

            mini_total_error = 0
            for pos, point in zip(self.Cam1_2.image_points1[i], self.Cam1_2.object_points1[i]):

                # 正解点のプロット
                if draw_flag:
                    _img = cv2.drawMarker(_img, tuple(pos[0]), color=(0, 255, 0), markerType=cv2.MARKER_SQUARE, thickness=1, line_type=cv2.LINE_4)
            
                # 予測点の変換
                # point2 = World2Camera(self.Cam1_2.rvecs1[i], self.Cam1_2.tvecs1[i], point)

                point = World2Camera(self.Cam1_2.rvecs2[Rt_idx], self.Cam1_2.tvecs2[Rt_idx], point)

                point = Camera2otherCamera(self.Cam1_2.R, self.Cam1_2.T, point)

                # print('正解 :', point2)
                # print('予測 :', point)
                # print('-'*30)

                if self.Cam1_2.fisheye:
                    point = calc_fisheye_distortion(self.Cam1_2.distortion1, point)
                else:
                    point = calc_distortion(self.Cam1_2.distortion1, point)

                point = Camera2Image(self.Cam1_2.camera_matrix1, point)

                # 予測点のプロット
                if draw_flag:
                    _point = list(map(int, point))
                    _img = cv2.drawMarker(_img, tuple(_point), color=(0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, thickness=1, line_type=cv2.LINE_4)
            

                # 誤差計算
                mini_total_error += np.linalg.norm(pos[0] - point)

            # 誤差計算
            mini_total_error = mini_total_error / len(self.Cam1_2.object_points1[i])


            if draw_flag:
                print(f'画像{i+1}の誤差合計 = {mini_total_error}\n')
                plt.imshow(_img)
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                plt.axis('off')
                plt.show()

            total_error += mini_total_error
            error_list.append(mini_total_error)

        total_error = total_error / len(self.Cam1_2.object_points1)
        print('全画像の誤差平均 =', total_error, '\n')

        if write:
            draw_error(error_list, title, output, total_error)

    def evaluate_Image2World(self, draw_flag=False, title='title', output='./', write=False):
        total_error = 0
        error_list = []

        for k in range(len(self.Cam1_2.object_points1)):
            if self.Rt1_num is not None:
                Rt1_idx = self.Rt1_num
                Rt2_idx = self.Rt2_num
            else:
                Rt1_idx = k
                Rt2_idx = k


            if draw_flag:
                _img = self.Cam1_2.imgs1[k]

            if self.Cam1_2.fisheye:
                undistorted_point = cv2.fisheye.undistortPoints(self.Cam1_2.image_points2[k], self.Cam1_2.camera_matrix2, self.Cam1_2.distortion2, P=self.Cam1_2.camera_matrix2)
            else:
                undistorted_point = cv2.undistortPoints(self.Cam1_2.image_points2[k], self.Cam1_2.camera_matrix2, self.Cam1_2.distortion2, P=self.Cam1_2.camera_matrix2)

            mini_total_error = 0
            l=1
            for i in range(len(self.Cam1_2.object_points1[k])):

                # 世界座標(cam1)の正解点
                w_p = self.Cam1_2.object_points1[k][i]

                # 画像座標(cam2)の対応点 -> カメラ座標(cam2) -> カメラ座標(cam1) -> 世界座標(cam1)に変換
                p = undistorted_point[i][0]
                # if (k+1 == 44 or k+1 == 45) and l:
                #     print(p)
                p = np.r_[p, [1]]
                p = Image2Camera(self.Cam1_2.camera_matrix2, p)
                # if (k+1 == 44 or k+1 == 45) and l:
                #     print(p)
                # if (k+1 == 38 or k+1 == 39) and l:
                #     l=0
                    # print(f'{k+1}:\n{self.Cam1_2.rvecs2[k]}\n\n{self.Cam1_2.tvecs2[k]}\n')
                    

                #     rmat, _ = cv2.Rodrigues(self.Cam1_2.rvecs2[k])
                #     t1, t2, t3 = self.Cam1_2.tvecs2[k].reshape(1, -1)[0]
                #     r13, r23, r33 = rmat[0][2], rmat[1][2], rmat[2][2]
                #     u, v, _ = p

                #     # Ax = bを解く
                #     A = np.array(
                #         [[1, 0, -u],
                #         [0, 1, -v],
                #         [r13, r23, r33]])

                #     print(f'A = {A}')

                #     print(f'inv A = {np.linalg.inv(A)}')

                #     print(f'b = {np.array([0, 0, (r13*t1 + r23*t2 + r33*t3)])}')

                p = homogeneous2point3d(p, self.Cam1_2.rvecs2[Rt2_idx], self.Cam1_2.tvecs2[Rt2_idx])
                # if (k+1 == 44 or k+1 == 45) and l:
                #     print(p)
                p = Camera2otherCamera(self.Cam1_2.R, self.Cam1_2.T, p)
                # if (k+1 == 44 or k+1 == 45) and l:
                #     print(p)
                p = Camera2World(self.Cam1_2.rvecs1[Rt1_idx], self.Cam1_2.tvecs1[Rt1_idx], p)
                # if (k+1 == 45 or k+1 == 44) and l:
                #     print(p);l=0

                # 誤差計算
                # print(w_p)
                # print(p)
                # print('-'*30)
                mini_total_error += np.linalg.norm(w_p - p)

                if draw_flag:
                    if self.Cam1_2.fisheye:
                        draw_w_p, _ = cv2.fisheye.projectPoints(w_p[np.newaxis, np.newaxis, :], self.Cam1_2.rvecs1[Rt1_idx], self.Cam1_2.tvecs1[Rt1_idx], self.Cam1_2.camera_matrix1, self.Cam1_2.distortion1)

                        draw_p, _ = cv2.fisheye.projectPoints(p[np.newaxis, np.newaxis, :], self.Cam1_2.rvecs1[Rt1_idx], self.Cam1_2.tvecs1[Rt1_idx], self.Cam1_2.camera_matrix1, self.Cam1_2.distortion1)
                    else:
                        draw_w_p, _ = cv2.projectPoints(w_p[np.newaxis, :], self.Cam1_2.rvecs1[Rt1_idx], self.Cam1_2.tvecs1[Rt1_idx], self.Cam1_2.camera_matrix1, self.Cam1_2.distortion1)

                        draw_p, _ = cv2.projectPoints(p[np.newaxis, :], self.Cam1_2.rvecs1[Rt1_idx], self.Cam1_2.tvecs1[Rt1_idx], self.Cam1_2.camera_matrix1, self.Cam1_2.distortion1)
                    
                    draw_w_p = list(map(int, draw_w_p[0][0]))
                    _img = cv2.drawMarker(_img, tuple(draw_w_p), color=(255, 0, 0), markerType=cv2.MARKER_SQUARE, thickness=1, line_type=cv2.LINE_4)

                    draw_p = list(map(int, draw_p[0][0]))
                    _img = cv2.drawMarker(_img, tuple(draw_p), color=(0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, thickness=1, line_type=cv2.LINE_4)


            # 誤差計算
            mini_total_error = mini_total_error / len(self.Cam1_2.object_points1[k])

            # print(k+1, self.Cam1_2.tvecs2[k][2])
            # print(f'画像{k+1}の誤差合計 = {mini_total_error}\n')

            if draw_flag:
            # if draw_flag and (k+1 == 44 or k+1 == 45):
                print(f'画像{k+1}の誤差合計 = {mini_total_error}\n')
                # print(self.Cam1_2.image_points2[k])
                plt.imshow(_img)
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                plt.axis('off')
                plt.show()

            total_error += mini_total_error
            error_list.append(mini_total_error)

        


        total_error = total_error / len(self.Cam1_2.object_points1)
        print('全画像の誤差平均 =', total_error, '\n')

        if write:
            draw_error(error_list, title, output, total_error)

def evaluate_3camTrans(Cam1_2, Cam2_3, draw_flag=False, title='title', output='./', write=False):
    '''
    ３カメラでチェッカーボードが全て映っている画像を利用した評価関数
    '''
    total_error = 0
    error_list = []

    for k in range(len(Cam1_2.object_points1)):

        if draw_flag:
            _img = Cam1_2.imgs1[k]

        undistorted_point = cv2.undistortPoints(Cam2_3.image_points2[k], Cam2_3.camera_matrix2, Cam2_3.distortion2, P=Cam2_3.camera_matrix2)

        mini_total_error = 0
        for i in range(len(Cam1_2.object_points1[k])):

            # 世界座標(cam1)の正解点
            w_p = Cam1_2.object_points1[k][i]

            # 画像座標(cam2)の対応点 -> カメラ座標(cam2) -> カメラ座標(cam1) -> 世界座標(cam1)に変換
            p = undistorted_point[i][0]
            p = np.r_[p, [1]]
            p = Image2Camera(Cam2_3.camera_matrix2, p)
            p = homogeneous2point3d(p, Cam2_3.rvecs2[k], Cam2_3.tvecs2[k])
            p = Camera2otherCamera(Cam2_3.R, Cam2_3.T, p)
            p = Camera2otherCamera(Cam1_2.R, Cam1_2.T, p)
            p = Camera2World(Cam1_2.rvecs1[k], Cam1_2.tvecs1[k], p)

            # 誤差計算
            mini_total_error += np.linalg.norm(w_p - p)

            if draw_flag:
                draw_w_p, _ = cv2.projectPoints(w_p[np.newaxis, :], Cam1_2.rvecs1[k], Cam1_2.tvecs1[k], Cam1_2.camera_matrix1, Cam1_2.distortion1)
                draw_w_p = list(map(int, draw_w_p[0][0]))
                _img = cv2.drawMarker(_img, tuple(draw_w_p), color=(255, 0, 0), markerType=cv2.MARKER_SQUARE, thickness=1, line_type=cv2.LINE_4)

                draw_p, _ = cv2.projectPoints(p[np.newaxis, :], Cam1_2.rvecs1[k], Cam1_2.tvecs1[k], Cam1_2.camera_matrix1, Cam1_2.distortion1)
                draw_p = list(map(int, draw_p[0][0]))
                _img = cv2.drawMarker(_img, tuple(draw_p), color=(0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, thickness=1, line_type=cv2.LINE_4)


        # 誤差計算
        mini_total_error = mini_total_error / len(Cam1_2.object_points1[k])

        if draw_flag:
            print(f'画像{k+1}の誤差合計 = {mini_total_error}\n')
            plt.imshow(_img)
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.axis('off')
            plt.show()

        total_error += mini_total_error
        error_list.append(mini_total_error)


    total_error = total_error / len(Cam1_2.object_points1)
    print('全画像の誤差平均 =', total_error, '\n')
    
    if write:
        draw_error(error_list, title, output, total_error)

def evaluate_2camTrans_invisible_ItoW(from_image_points, to_world_points, from_K, from_D, from_rvec, from_tvec, to_K, to_D, to_rvec, to_tvec, c_img, R, T, draw_flag=False):
    total_error = 0
    error_list = []

    undistorted_point = cv2.undistortPoints(from_image_points, from_K, from_D, P=from_K)

    if draw_flag:
        img = c_img.copy()

    for i in range(len(to_world_points)):

        # 世界座標(cam1)の正解点
        w_p = to_world_points[i]

        # 画像座標(cam2)の対応点 -> カメラ座標(cam2) -> カメラ座標(cam1) -> 世界座標(cam1)に変換
        p = undistorted_point[i][0]
        p = np.r_[p, [1]]
        p = Image2Camera(from_K, p)
        p = homogeneous2point3d(p, from_rvec, from_tvec)
        pp = Camera2World(from_rvec, from_tvec, p)
        p = Camera2otherCamera(R, T, p)
        p = Camera2World(to_rvec, to_tvec, p)

        if draw_flag:
            draw_w_p, _ = cv2.projectPoints(w_p[np.newaxis, :], to_rvec, to_tvec, to_K, to_D)
            draw_w_p = list(map(int, draw_w_p[0][0]))
            img = cv2.drawMarker(img, tuple(draw_w_p), color=(255, 0, 0), markerType=cv2.MARKER_SQUARE, thickness=1, line_type=cv2.LINE_4)

            draw_p, _ = cv2.projectPoints(p[np.newaxis, :], to_rvec, to_tvec, to_K, to_D)
            draw_p = list(map(int, draw_p[0][0]))
            img = cv2.drawMarker(img, tuple(draw_p), color=(0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, thickness=1, line_type=cv2.LINE_4)

            print(f'w_p = {w_p}')
            print(f'pp = {pp}')
            print(f'p = {p}\n---\n')


        # 誤差計算
        total_error += np.linalg.norm(w_p - p)

    print(f'誤差合計 = {total_error}\n')

    if draw_flag:
        plt.imshow(img)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.axis('off')
        plt.show()




    



# 以下，ミニ関数

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

def Camera2World(rvec, tvec, p):
    '''
    input:
    rvec = [[x], [y], [z]] : 回転ベクトル
    tvec = [[x], [y], [z]] : 並進ベクトル
    p = [x, y, z] : 対象点

    output:
    trans_p = [x, y, z] : 移動後の点
    '''

    rvec = rvec.reshape(1, -1)[0] * -1
    tvec = tvec.reshape(1, -1)[0]
    rvec_ = rvec / np.linalg.norm(rvec, ord=2)
    theta = np.linalg.norm(rvec, ord=2)
    trans_p = p - tvec
    trans_p = trans_p*np.cos(theta) + rvec_*np.dot(rvec_, trans_p)*(1-np.cos(theta)) + np.cross(rvec_, trans_p)*np.sin(theta)

    return trans_p

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

def calc_distortion(distortion, p):
    '''
    レンズ歪みを復元する関数
    '''
    k1, k2, p1, p2, k3 = distortion[0]
    x, y, z = p
    x = x / z
    y = y / z
    r_2 = x**2 + y**2
    x_ = x*(1 + k1*r_2 + k2*(r_2**2) + k3*(r_2**3)) + 2*p1*x*y + p2*(r_2 + (2*(x**2)))
    y_ = y*(1 + k1*r_2 + k2*(r_2**2) + k3*(r_2**3)) + p1*(r_2 + (2*(y**2))) + 2*p2*x*y

    return (x_, y_)

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

def Camera2Image_isskew(cameraMatrix, p):
    '''
    内部パラメータを用いて カメラ座標 -> 画像座標 の変換を計算する関数
    '''
    f_x, f_y, c_x, c_y, skew = cameraMatrix[0][0], cameraMatrix[1][1], cameraMatrix[0][2], cameraMatrix[1][2], cameraMatrix[0][1]
    x, y = p
    x = f_x*(x + (skew*y)) + c_x
    y = f_y*y + c_y

    return (x, y)

def Image2Camera(cameraMatrix, p):
    '''
    内部パラメータを用いて 画像座標 -> カメラ座標 の変換を計算する関数
    '''

    return np.dot(np.linalg.inv(cameraMatrix), p)

def conv_points(p, H):
    '''
    ホモグラフィ行列を用いて点を変換する関数
    '''

    p = np.r_[p[0], [1]]
    p = np.dot(H, p)
    p = (int(p[0] / p[2]), int(p[1] / p[2]))

    return p

def homogeneous2point3d(homo_p, rvec, tvec):
    '''
    同次座標系(2D) -> 通常の座標系(3D)を，世界座標が z = 0 であることを利用して解く関数

    Input:
    homo_p = (u, v, 1) : 同次座標系の点
    rvec : 回転ベクトル
    tvec : 並進ベクトル

    Output:
    trans_p = (x, y, z) : 通常座標の点
    '''

    if homo_p[2] != 1:  # 同次座標(z = 1)でないならエラー
        print('homo error');exit()

    rmat, _ = cv2.Rodrigues(rvec)   # 回転ベクトル -> 回転行列
    t1, t2, t3 = tvec.reshape(1, -1)[0]
    r13, r23, r33 = rmat[0][2], rmat[1][2], rmat[2][2]
    u, v, _ = homo_p

    # Ax = bを解く
    A = np.array(
        [[1, 0, -u],
        [0, 1, -v],
        [r13, r23, r33]])

    b = np.array([0, 0, (r13*t1 + r23*t2 + r33*t3)])

    trans_p = np.dot(np.linalg.inv(A), b)

    return trans_p

def point2D2point3D(c_object_points, camera_matrix_1, camera_matrix_2, distortion_1, distortion_2, rvecs_1, rvecs_2, tvecs_1, tvecs_2):
    '''
    キャリブレーション結果を用いて 世界座標点 -> 画像座標 に変換する関数
    '''
    
    point2D_1s_row = None
    point2D_2s_row = None
    point2D_1s_col = []
    point2D_2s_col = []
    for i in range(len(c_object_points)):
        point2D_1, _ = cv2.projectPoints(c_object_points[i], rvecs_1[i], tvecs_1[i], camera_matrix_1, distortion_1)
        point2D_2, _ = cv2.projectPoints(c_object_points[i], rvecs_2[i], tvecs_2[i], camera_matrix_2, distortion_2)
        if i < len(c_object_points) - 2: # 2枚を推定用にする
            if point2D_1s_row is None:
                point2D_1s_row = point2D_1
                point2D_2s_row = point2D_2
            else:
                point2D_1s_row = np.r_[point2D_1s_row, point2D_1]
                point2D_2s_row = np.r_[point2D_2s_row, point2D_2]
        point2D_1s_col.append(point2D_1)
        point2D_2s_col.append(point2D_2)

    return point2D_1s_row, point2D_2s_row, point2D_1s_col, point2D_2s_col

def conv_and_draw_by_homography(c_imgs_cam1, c_image_points_cam1, c_object_points, camera_matrix_1, camera_matrix_2, distortion_1, distortion_2, rvecs_1, rvecs_2, tvecs_1, tvecs_2, draw_flag=False):
    '''
    ホモグラフィ変換を用いてカメラ161の点をカメラ159の点に変換し, 描写する関数
    '''
    # 点の変換
    point2D_1s_row, point2D_2s_row, point2D_1s_col, point2D_2s_col = point2D2point3D(c_object_points, camera_matrix_1, camera_matrix_2, distortion_1, distortion_2, rvecs_1, rvecs_2, tvecs_1, tvecs_2)

    H, _ = cv2.findHomography(point2D_2s_row, point2D_1s_row, cv2.RANSAC, 5.0)

    for i in range(len(c_object_points)):
        if i < len(c_object_points) - 4:  # ホモグラフィ行列予測に使った画像２枚と使ってない２枚のみプロットする(つまり後半の４枚)
            continue
        _img = c_imgs_cam1[i]


        point2D_1 = point2D_1s_col[i]
        point2D_2 = point2D_2s_col[i]
        for pos, point in zip(c_image_points_cam1[i], point2D_2):
            
            # 正解点のプロット
            if draw_flag:
                _img = cv2.drawMarker(_img, tuple(pos[0]), color=(0, 255, 0), markerType=cv2.MARKER_SQUARE, thickness=1, line_type=cv2.LINE_4)

            # 予測点の変換
            point = conv_points(point, H)

            # 予測点のプロット
            if draw_flag:
                _img = cv2.drawMarker(_img, tuple(point), color=(0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, thickness=1, line_type=cv2.LINE_4)
        
        if draw_flag:
            plt.imshow(_img)
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.axis('off')
            plt.show()



def draw_error(error_list, title, output, total_error):

    plt.figure()
    x = list(range(len(error_list)))
    plt.bar(x, error_list, width=0.9, align='center')
    plt.title(title)
    plt.xlabel(f'全画像の誤差平均 = {total_error}')
    plt.ylabel('reprojection error')
    plt.savefig(output + title + '.png')