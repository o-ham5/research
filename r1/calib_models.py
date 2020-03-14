import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import time
# from sklearn.svm import SVR


class Single_Camera:
    
    def __init__(self, paths, cols, rows, chess_size, fisheye=False):
        self.paths = paths
        self.cols = cols
        self.rows = rows
        self.chess_size = chess_size
        self.fisheye = fisheye

    def read_file(self):
        imgs = []
        for path in self.paths:
            img = cv2.imread(path)
            imgs.append(img)
        
        self.imgs_all = imgs

    def find_chessCorner(self):
        image_points = []
        imgs_main = []
        for k, img in enumerate(self.imgs_all):
            # 画像をグレースケールに変換
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # チェスボードの交点を検出
            ret, corners = cv2.findChessboardCorners(gray, (self.cols, self.rows))

            if ret:  # すべての交点の検出に成功
                image_points.append(corners)
                imgs_main.append(img)
            else:
                print(f'image{k+1} corners detection failed.')
    
        image_points = np.array(image_points, dtype=np.float32)
        self.image_points = image_points
        self.imgs = imgs_main

    def sample(self, img, image_point, ret=True):
        cv2.drawChessboardCorners(img, (self.cols, self.rows), image_point, ret)
        plt.imshow(img)
        plt.show()

    def make_worldPoint(self):
        world_points = np.zeros((self.rows * self.cols, 3), np.float32)
        world_points[:, :2] = np.mgrid[:self.cols, :self.rows].T.reshape(-1, 2) * self.chess_size

        object_points = np.array([world_points] * len(self.image_points), dtype=np.float32)

        self.object_points = object_points

    def calibration(self):
        
        if self.fisheye:
            self.fisheye_calibration()
        else:
            self.box_calibration()

    def box_calibration(self):
        h, w, _ = self.imgs[0].shape
        ret, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points, self.image_points, (w, h), None, None)

        self.reprojection_error = ret
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.rvecs = rvecs
        self.tvecs = tvecs

    def fisheye_calibration(self):
        h, w, _ = self.imgs[0].shape
        N = len(self.object_points)
        # K = np.eye(3, dtype=np.float64)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N)]
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

        # object_points.shape = (<num of calibration images>, 1, <num points in set>, 3)にする
        object_points = self.object_points[:, np.newaxis]

        ret, camera_matrix, distortion, rvecs, tvecs = cv2.fisheye.calibrate(
            object_points, self.image_points, (w, h), K, D, rvecs, tvecs, calibration_flags)



        self.reprojection_error = ret
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.rvecs = rvecs
        self.tvecs = tvecs

    def execusion_all(self):

        ##### 1. チェスボードを撮影した画像を読み込む #####

        print('--- 1 ---\n')
        ts_1 = time()

        self.read_file()
        print('カメラの画像数 :', len(self.imgs_all), '\n')

        te_1 = time()

        ##### 2. チェスボードの交点検出 #####

        print('--- 2 ---\n')
        ts_2 = time()

        # チェスボードのマーカー検出を行う。
        self.find_chessCorner()
        print('使用する画像数 :', len(self.imgs), '\n')

        te_2 = time()

        ##### 3. 検出した画像座標上の点に対応する3次元上の点を作成する #####

        print('--- 3 ---\n')

        self.make_worldPoint()

        #####  4. キャリブレーション  #####

        print('--- 4 ---\n')

        ts_4 = time()

        self.calibration()
        print('再投影誤差 =\n', self.reprojection_error, '\n')
        print('カメラ行列 =\n', self.camera_matrix, '\n')

        te_4 = time()

        print('---\nプロファイル\n')

        print('画像読み込み :', te_1 - ts_1, '[s]')
        print('交点検出 :', te_2 - ts_2, '[s]')
        print('キャリブレーション :', te_4 - ts_4, '[s]')

    def estimate_extrinsicParams(self, camera_matrix_2, distortion_2):
        rvecs = []
        tvecs = []
        # if self.fisheye:
        #     h, w, _ = self.imgs[0].shape
        #     N = len(self.object_points)
        #     rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N)]
        #     tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N)]
        #     object_points = self.object_points[:, np.newaxis]
        #     calibration_flags = cv2.fisheye.CALIB_FIX_INTRINSIC
            
        #     camera_matrix_2_copy = np.copy(camera_matrix_2)
        #     distortion_2_copy = np.copy(distortion_2)

        #     _, cam, _, rvecs, tvecs = cv2.fisheye.calibrate(
        #     object_points, self.image_points, (w, h), camera_matrix_2_copy, distortion_2_copy, rvecs, tvecs, calibration_flags)

        # else:
        for o_point, i_point in zip(self.object_points, self.image_points):
            rvec=np.zeros((3, 1))
            tvec=np.zeros((3, 1))
            # o_point = np.array([o_point[0], o_point[3], o_point[7], o_point[10]])
            # i_point = np.array([i_point[0], i_point[3], i_point[7], i_point[10]])
            if self.fisheye:
                # _, rvec, tvec, _ = cv2.solvePnPRansac(o_point, i_point, camera_matrix_2, distortion_2, rvec, tvec, False, 100, 8.0, 0.99, np.array([]), cv2.SOLVEPNP_EPNP, self.fisheye)
                ret, rvec, tvec = cv2.solvePnP(o_point, i_point, camera_matrix_2, distortion_2, rvec, tvec, False, cv2.SOLVEPNP_EPNP, self.fisheye)
            else:
                # _, rvec, tvec, _ = cv2.solvePnPRansac(o_point, i_point, camera_matrix_2, distortion_2, rvec, tvec, False, 100, 8.0, 0.99, np.array([]), cv2.SOLVEPNP_ITERATIVE)
                ret, rvec, tvec = cv2.solvePnP(o_point, i_point, camera_matrix_2, distortion_2, rvec, tvec, False, cv2.SOLVEPNP_ITERATIVE, self.fisheye)
        
            rvecs.append(rvec)
            tvecs.append(tvec)

        return rvecs, tvecs
        
    def fish_estimate_extrinsicParams_by_MLS(self, camera_matrix_2, distortion_2):

        cam_1, cam_2, cam_3 = [], [], []
        wor = []

        for k in range(len(self.object_points)):

            undistorted_point = cv2.fisheye.undistortPoints(self.image_points[k], self.camera_matrix, self.distortion, P=self.camera_matrix)
    
            w_ps = []
            c_ps = []
            for i in range(len(self.object_points[k])):

                # 世界座標の正解点
                w_p = self.object_points[k][i]


                # 画像座標の正解点(対応点)を世界座標に変換
                p = undistorted_point[i][0]
                p = np.r_[p, [1]]
                p = Image2Camera(self.Cam.camera_matrix, p)

                w_ps.append(w_p)
                c_ps.append(p)

            model = SVR()


        return rvecs, tvecs

    def show_unditortImage(self, img):
        img_undistorted = cv2.undistort(img, self.camera_matrix, self.distortion)
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(img_undistorted)
        plt.show()

    def show_fisheye_undistortImage(self, img):
        h, w, _ = img.shape
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.camera_matrix, self.distortion, np.eye(3), self.camera_matrix, (w, h), cv2.CV_16SC2)
        img_undistorted = cv2.remap(img, map1, map2, interpolation=cv2.BORDER_CONSTANT)
        # img_undistorted = cv2.fisheye.undistortImage(img,
        # self.camera_matrix, self.distortion, self.camera_matrix, (w, h))
        plt.figure()
        plt.imshow(img)
        plt.show()
        plt.figure()
        plt.imshow(img_undistorted)
        plt.show()

class Stereo_Camera:
    def __init__(self, paths1, paths2, cols1, cols2, rows1, rows2, chess_size1, chess_size2, camera_matrix1, camera_matrix2, distortion1, distortion2, fisheye=False):
        self.paths1 = paths1
        self.cols1 = cols1
        self.rows1 = rows1
        self.chess_size1 = chess_size1
        self.camera_matrix1 = camera_matrix1
        self.distortion1 = distortion1

        self.paths2 = paths2
        self.cols2 = cols2
        self.rows2 = rows2
        self.chess_size2 = chess_size2
        self.camera_matrix2 = camera_matrix2
        self.distortion2 = distortion2

        self.fisheye = fisheye

    def read_file(self):
        imgs1 = []
        for path in self.paths1:
            img = cv2.imread(path)
            imgs1.append(img)
        

        imgs2 = []
        for path in self.paths2:
            img = cv2.imread(path)
            imgs2.append(img)
        
        self.imgs1_all = imgs1
        self.imgs2_all = imgs2

    def find_chessCorner(self):
        image_points1 = []
        imgs1_main = []
        image_points2 = []
        imgs2_main = []

        for k, (img_1, img_2) in enumerate(zip(self.imgs1_all, self.imgs2_all)):
            # 画像をグレースケールに変換する。
            gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
            # チェスボードの交点を検出する。
            ret_1, corners_1 = cv2.findChessboardCorners(gray_1, (self.cols1, self.rows2))
            ret_2, corners_2 = cv2.findChessboardCorners(gray_2, (self.cols2, self.rows2))

            if ret_1 and ret_2:  # 2画像のすべての交点の検出に成功
                image_points1.append(corners_1)
                imgs1_main.append(img_1)
                image_points2.append(corners_2)
                imgs2_main.append(img_2)

            else:
                print(f'image{k+1} corners detection failed.')
    
        image_points1 = np.array(image_points1, dtype=np.float32)
        self.image_points1 = image_points1
        self.imgs1 = imgs1_main

        image_points2 = np.array(image_points2, dtype=np.float32)
        self.image_points2 = image_points2
        self.imgs2 = imgs2_main

    def make_worldPoint(self):
        world_points1 = np.zeros((self.rows1 * self.cols1, 3), np.float32)
        world_points1[:, :2] = np.mgrid[:self.cols1, :self.rows1].T.reshape(-1, 2) * self.chess_size1

        object_points1 = np.array([world_points1] * len(self.image_points1), dtype=np.float32)

        self.object_points1 = object_points1

        world_points2 = np.zeros((self.rows2 * self.cols2, 3), np.float32)
        world_points2[:, :2] = np.mgrid[:self.cols2, :self.rows2].T.reshape(-1, 2) * self.chess_size2

        object_points2 = np.array([world_points2] * len(self.image_points2), dtype=np.float32)

        self.object_points2 = object_points2

    def stereo_calibration(self):
        if self.fisheye:
            self.fisheye_stereo_calibration()
        else:
            self.box_stereo_calibration()

    def box_stereo_calibration(self):
        '''
        2 -> 1
        '''
        h, w, _ = self.imgs1[0].shape
        ret, camera_matrix2, distortion2, camera_matrix1, distortion1, R, T, E, F = cv2.stereoCalibrate(self.object_points1, self.image_points2, self.image_points1, self.camera_matrix2, self.distortion2, self.camera_matrix1, self.distortion1, (w, h), flags=cv2.CALIB_FIX_INTRINSIC)

        self.projection_error = ret
        self.R = R
        self.T = T
        self.E = E
        self.F = F

    def fisheye_stereo_calibration(self):
        '''
        2 -> 1
        '''
        h, w, _ = self.imgs1[0].shape
        N = len(self.object_points1)
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N)]
        R = np.zeros((1, 1, 3), dtype=np.float64)
        T = np.zeros((1, 1, 3), dtype=np.float64)
        calibration_flags = cv2.fisheye.CALIB_FIX_INTRINSIC
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 1e-6)

        # object_points.shape = (<num of calibration images>, 1, <num points in set>, 3)にする
        object_points = self.object_points1[:, np.newaxis]

        # image_points.shape = (<num of calibration images>, 1, <num points in set>, 2)にする
        shape1 = self.image_points1.shape
        shape2 = self.image_points2.shape
        image_points1 = np.reshape(self.image_points1, (shape1[0], shape1[2], shape1[1], shape1[3]))
        image_points2 = np.reshape(self.image_points2, (shape2[0], shape2[2], shape2[1], shape2[3]))


        ret, camera_matrix2, distortion2, camera_matrix1, distortion1, R, T = cv2.fisheye.stereoCalibrate(object_points, image_points2, image_points1, self.camera_matrix2, self.distortion2, self.camera_matrix1, self.distortion1, (w, h), R, T, calibration_flags)

        self.projection_error = ret
        self.R = R
        self.T = T

    def estimate_extrinsicParams(self):
        rvecs1 = []
        tvecs1 = []
        for o_point, i_point in zip(self.object_points1, self.image_points1):
            rvec=np.zeros((3, 1))
            tvec=np.zeros((3, 1))
            if self.fisheye:
                ret, rvec, tvec = cv2.solvePnP(o_point, i_point, self.camera_matrix1, self.distortion1, rvec, tvec, False, cv2.SOLVEPNP_EPNP, self.fisheye)
                # _, rvec, tvec, _ = cv2.solvePnPRansac(o_point, i_point, self.camera_matrix1, self.distortion1, rvec, tvec, False, 100, 8.0, 0.99, np.array([]), cv2.SOLVEPNP_EPNP, self.fisheye)
            else:
                ret, rvec, tvec = cv2.solvePnP(o_point, i_point, self.camera_matrix1, self.distortion1, rvec, tvec, False, cv2.SOLVEPNP_ITERATIVE, self.fisheye)
                # _, rvec, tvec, _ = cv2.solvePnPRansac(o_point, i_point, self.camera_matrix1, self.distortion1, rvec, tvec, False, 100, 8.0, 0.99, np.array([]), cv2.SOLVEPNP_ITERATIVE, self.fisheye)
            rvecs1.append(rvec)
            tvecs1.append(tvec)

        rvecs2 = []
        tvecs2 = []
        for o_point, i_point in zip(self.object_points2, self.image_points2):
            rvec=np.zeros((3, 1))
            tvec=np.zeros((3, 1))
            if self.fisheye:
                ret, rvec, tvec = cv2.solvePnP(o_point, i_point, self.camera_matrix2, self.distortion2, rvec, tvec, False, cv2.SOLVEPNP_EPNP, self.fisheye)
                # _, rvec, tvec, _ = cv2.solvePnPRansac(o_point, i_point, self.camera_matrix2, self.distortion2, rvec, tvec, False, 100, 8.0, 0.99, np.array([]), cv2.SOLVEPNP_EPNP, self.fisheye)
            else:
                ret, rvec, tvec = cv2.solvePnP(o_point, i_point, self.camera_matrix2, self.distortion2, rvec, tvec, False, cv2.SOLVEPNP_ITERATIVE, self.fisheye)
                # _, rvec, tvec, _ = cv2.solvePnPRansac(o_point, i_point, self.camera_matrix2, self.distortion2, rvec, tvec, False, 100, 8.0, 0.99, np.array([]), cv2.SOLVEPNP_ITERATIVE, self.fisheye)

            rvecs2.append(rvec)
            tvecs2.append(tvec)

        self.rvecs1 = rvecs1
        self.tvecs1 = tvecs1
        self.rvecs2 = rvecs2
        self.tvecs2 = tvecs2

        return rvecs1, tvecs1, rvecs2, tvecs2

    def fish_estimate_extrinsicParams(self):
        h, w, _ = self.imgs1[0].shape
        N = len(self.object_points1)
        K1 = np.copy(self.camera_matrix1)
        D1 = np.copy(self.distortion1)
        K2 = np.copy(self.camera_matrix2)
        D2 = np.copy(self.distortion2)
        rvecs1 = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N)]
        tvecs1 = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N)]
        rvecs2 = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N)]
        tvecs2 = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N)]
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

        # object_points.shape = (<num of calibration images>, 1, <num points in set>, 3)にする
        object_points = self.object_points1[:, np.newaxis]

        ret1, camera_matrix1, distortion1, rvecs1, tvecs1 = cv2.fisheye.calibrate(
            object_points, self.image_points1, (w, h), K1, D1, rvecs1, tvecs1, calibration_flags)

        ret2, camera_matrix2, distortion2, rvecs2, tvecs2 = cv2.fisheye.calibrate(
            object_points, self.image_points2, (w, h), K2, D2, rvecs2, tvecs2, calibration_flags)

        self.rvecs1 = rvecs1
        self.tvecs1 = tvecs1
        self.rvecs2 = rvecs2
        self.tvecs2 = tvecs2

        return rvecs1, tvecs1, rvecs2, tvecs2



