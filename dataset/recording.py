import os
import numpy as np
import cv2
import json
from tqdm import tqdm
import io
import zstandard as zstd
from PIL import Image
import glob

#from dataset.base_dataset import BaseDataset
from utils.constants import VIT_JOINTS_NAME
from utils.calib_utils import undistort_cameras_from_json




class Recording:
    def __init__(self, root_dir, path_cam_meta, logger, recording_name='', cam_keys=None, load_undistort_images=False):
        #super(Recording, self).__init__(logger)

        self.logger = logger

        self.data_tag='egoexo'
        self.root_dir= root_dir
        self.load_undistort_images = load_undistort_images
        self.recording_name = recording_name
        self.cam_keys = cam_keys

        self.input_zup = False

        self.joint_set = {
            'name': 'EgoExo',
            'joint_num': 19, #add a verbose neck and pelvis
            'joints_name': ('Nose', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip',  'L_Knee', 'L_Ankle', 'R_Eye', 'L_Eye', 'R_Ear',  'L_Ear', 'Neck', 'Pelvis'),
            'original_joints_name' : ('nose', 'right-shoulder','right-elbow', 'right-wrist', 'left-shoulder','left-elbow','left-wrist', 'right-hip', 'right-knee', 'right-ankle', 'left-hip','left-knee','left-ankle', 'right-eye', 'left-eye','right-ear', 'left-ear', 'neck', 'pelvis'),
            'flip_pairs': ((1, 4), (2, 5), (3, 6), (7, 10), (8, 11), (9, 12), (13, 14), (15, 16)),
            'skeleton': ((1,2),(2,3),(1,7),(7,8),(8,9),(4,5),(5,6),(4,10),(10,11),(11,12),(0,13),(13,15),(0,14),(14,16),(0,17),(17,18),(1,17),(4,17),(7,18),(10,18)),
        }
        self.r_shd_idx = self.joint_set['joints_name'].index('R_Shoulder')
        self.l_shd_idx = self.joint_set['joints_name'].index('L_Shoulder')
        self.r_hip_idx = self.joint_set['joints_name'].index('R_Hip')
        self.l_hip_idx = self.joint_set['joints_name'].index('L_Hip')
        self.neck_idx = self.joint_set['joints_name'].index('Neck')
        self.pelvis_idx = self.joint_set['joints_name'].index('Pelvis')

        self.image_hw = (2160, 3840) 

        self.cam_info = self.load_camera_meta(path_cam_meta)
        self.datalist = self.load_data()

    
    def load_camera_meta(self, path_cam_meta):        
        # load dict from json
        cam_meta, json_intrinsics_undist = undistort_cameras_from_json(path_cam_meta)

        cam_intrs = cam_meta['cameras']
        cam_intrs_undist = json_intrinsics_undist['cameras']

        if 'camera_world2cam' in cam_meta:
            cam_extrs = cam_meta['camera_world2cam']
            self.input_zup = True
        else:
            cam_extrs = cam_meta['camera_base2cam']
            self.logger.info("No 'camera_world2cam' in cam_meta, using 'camera_base2cam' instead.")
        
        cam_info = {}

        for cam_key in cam_intrs:
            cam_info[cam_key]={}
            cam_info[cam_key]['cam_intrinsics_ori'] = np.array(cam_intrs[cam_key]['K']).reshape(3,3)
            cam_info[cam_key]['dist'] = np.array(cam_intrs[cam_key]['dist']).flatten()
            cam_info[cam_key]['cam_intrinsics'] = np.array(cam_intrs_undist[cam_key]['K']).reshape(3,3)
        
        for cam_key in cam_extrs:
            cam_key2 = cam_key.split('_')[0]
            base2cam = np.eye(4)
            base2cam[:3, :3] = np.array(cam_extrs[cam_key]['R']).reshape(3,3)
            base2cam[:3, 3] = np.array(cam_extrs[cam_key]['T']).reshape(3,)
            cam_info[cam_key2]['cam_extrinsics'] = base2cam

        return cam_info
    



    def get_camera_params(self):
        dict_cam_params = {}
        for ccam_tag, ccam_info in self.cam_info.items():
            K = ccam_info['cam_intrinsics'].copy()
            focal = np.array([K[0,0],K[1,1]],dtype=np.float32)
            princpt = np.array([K[0,2],K[1,2]],dtype=np.float32)

            T_w2c = ccam_info['cam_extrinsics'].copy()

            # Extract rotation and translation from T_w2c
            R_w2c = T_w2c[:3, :3]  # 3x3 rotation matrix
            t_w2c = T_w2c[:3, 3]   # 3x1 translation vector

            # Convert rotation matrix to rotation vector
            rvec, _ = cv2.Rodrigues(R_w2c)
            tvec = t_w2c.reshape(-1, 1)

            dict_cam_params[ccam_tag] = {
                'K': K,
                'rvec': rvec.flatten(),
                'tvec': tvec.flatten(),
                'focal': focal,
                'princpt': princpt,
                'T_w2c': T_w2c,}

        
        return dict_cam_params

        

    def load_data(self):
        datalist = []
        ann_id = 0
        for cam_key in self.cam_info.keys():
            if self.cam_keys is not None and cam_key not in self.cam_keys:
                continue

            self.logger.info(f"Loading data for camera: {cam_key}")
            list_images = glob.glob(os.path.join(self.root_dir, cam_key, '*.png'))
            # sort list_images based on the frame number
            list_images = sorted(list_images, key=lambda x: int(x.split('_')[-1][:-4]))

            cam_intr = self.cam_info[cam_key]['cam_intrinsics']# here use the undistorted one
            cam_param = {'focal': np.array([cam_intr[0,0], cam_intr[1,1]], dtype=np.float32),
                        'princpt': np.array([cam_intr[0,2], cam_intr[1,2]], dtype=np.float32)}
            

            T_w2c = self.cam_info[cam_key]['cam_extrinsics']
            cam_param['T_w2c'] = T_w2c

            cseq_name = f"{self.recording_name}_{cam_key}"
        
            for cimg_path in tqdm(list_images):
                image_id = int(cimg_path.split('/')[-1].split('_')[-1][:-4])
                #if image_id<5000:
                #    continue
                # get image_path relative to self.root_dir
                image_path = os.path.relpath(cimg_path, self.root_dir)

                datalist.append({
                    'ann_id': ann_id,
                    'img_id': image_id,
                    'image_path': image_path,
                    'cam_param': cam_param,
                    'seq_name': cseq_name,
                })
                ann_id += 1

        print(f'Loaded {len(datalist)} images from {self.root_dir}')
        return datalist

    def get_image_from_lmdb(self, image_path):
        assert False, "lmdb loading is disabled for now, use cv2.imread instead."
        take_name, cam_key, image_tag = image_path.split("/")[-3:]
        image_tag = image_tag[:-4]

        print(f"Loading image from lmdb: {take_name}_{cam_key}, {image_tag}")

        subdb_img = self.env_img.open_db((f"{take_name}_{cam_key}").encode('ascii'),create=False)
        txn_img = self.env_img.begin(db=subdb_img,write=False)

        zstd_compressed_data = txn_img.get(image_tag.encode('ascii'))
        zstd_decompressor = zstd.ZstdDecompressor()
        webp_data = zstd_decompressor.decompress(zstd_compressed_data)
    
        # Convert WebP back to image
        image = Image.open(io.BytesIO(webp_data))
        img_rgb = np.array(image)
        return img_rgb
    
    def load_bgr_image_and_undistort(self, image_path):
        img_bgr = cv2.imread(os.path.join(self.root_dir, image_path))
        if not self.load_undistort_images:
            cam_key = image_path.split('/')[-2]
            ori_K = self.cam_info[cam_key]['cam_intrinsics_ori']
            ori_dist = self.cam_info[cam_key]['dist']
            new_K = self.cam_info[cam_key]['cam_intrinsics']
            img_bgr = cv2.undistort(img_bgr, ori_K, ori_dist, None, new_K)
        else:
            assert False

        return img_bgr




