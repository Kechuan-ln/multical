import os
import numpy as np
import cv2
import json
from tqdm import tqdm
import io
import zstandard as zstd
from PIL import Image

from dataset.base_dataset import BaseDataset
from utils.funcs_utils import transform_joint_to_other_db
from utils.constants import VIT_JOINTS_NAME



class EgoExo(BaseDataset):
    def __init__(self, root_dir, participant_uid, cam_keys, logger, load_data=True):
        super(EgoExo, self).__init__(logger)

        self.data_tag='egoexo'
        self.root_dir= root_dir

        self.participant_uid = participant_uid
        self.cam_keys = cam_keys#['cam03','cam02','cam04','cam01']#['cam01','cam02','cam03','cam04']


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

        self.root_lmdb = os.path.join('/media/fylwen/5a69d379-48b2-436d-9d41-84efe99a333d/egoexo/','lmdbs',f'lmdb_{participant_uid}')#os.path.join(self.root_dir, "lmdb")
        self.use_lmdb=True

        #set openpose
        self.use_gt2D = True
        self.openpose_threshold = 0.3
        self.root_openpose = os.path.join(self.root_dir, "annotations", "openpose")

        #annotations/ego_pose 
        self.root_poses_train = os.path.join(self.root_dir, "annotations", "ego_pose", "train", "body")
        self.root_poses_val = os.path.join(self.root_dir, "annotations", "ego_pose", "val", "body")
        self.root_camera_train = os.path.join(self.root_dir, "annotations", "ego_pose", "train", "camera_pose")
        self.root_camera_val = os.path.join(self.root_dir, "annotations", "ego_pose", "val", "camera_pose")

        
        pseudo_annotated_takes = os.listdir(os.path.join(self.root_poses_val,"automatic")) + os.listdir(os.path.join(self.root_poses_train,"automatic"))
        pseudo_annotated_takes = [take.split(".")[0] for take in pseudo_annotated_takes]        
        self.takes_uids = pseudo_annotated_takes
            
        #takes.json
        metadata = json.load(open(os.path.join(self.root_dir,"takes.json")))
        self.takes_metadata = {}

        #valid washed list
        self.list_valid_take_name_cam = []
        with open(f"../washed_trainval.txt", 'r') as f:
            for line in f:
                self.list_valid_take_name_cam.append(line.strip())                

        #metadata from takes.json
        for take_uid in self.takes_uids:
            take_temp = self.get_metadata_take(take_uid, metadata)
            if not take_temp["parent_task_name"] in ["Basketball","Soccer",'Dance','Bouldering','Rock Climbing']:
                continue
                
            if take_temp["participant_uid"] !=  self.participant_uid:
                continue


            if take_temp["take_name"] in ['unc_basketball_03-31-23_02_23','cmu_soccer06_4']:# pass this take, because it has a different subject other than participant_uid=388.
                print("pass", take_temp["take_name"])
                continue

            if take_temp['take_name'] not in ['uniandes_basketball_003_49']:
                print('pass', take_temp['take_name'])
                continue


            if take_temp and 'bouldering' not in take_temp['take_name']:
                self.takes_metadata[take_uid] = take_temp
    
        #sort self.takes_metadata by take_name
        self.takes_metadata = dict(sorted(self.takes_metadata.items(), key=lambda item: item[0]))

        if load_data:         
            self.datalist =  self.load_data()


    def get_metadata_take(self, uid, metadata):
        for take in metadata:
            if take["take_uid"]==uid:
                return take
            
    def get_list_seq_cam(self):
        list_take_cams=[]
        for cam_key in self.cam_keys:
            for take_uid in self.takes_metadata:
                take_name=self.takes_metadata[take_uid]["take_name"]
                if f"{take_name}_{cam_key}" in self.list_valid_take_name_cam:
                    list_take_cams.append(f"{take_name}_{cam_key}")
                
        return list_take_cams


    def load_data(self):
        datalist, seq_names  = [], []
        seq_start_end_idx = {}

        list_take_cams=[]
        for cam_key in self.cam_keys:
            for take_uid in self.takes_metadata:
                take_name=self.takes_metadata[take_uid]["take_name"]
                if f"{take_name}_{cam_key}" in self.list_valid_take_name_cam:
                    list_take_cams.append((take_uid, cam_key))
        
        
        aid=0
        for perm_idx, (take_uid, cam_key) in enumerate(list_take_cams):
            path_camera_json= os.path.join(self.root_camera_train, take_uid+".json")
            path_pose_json = os.path.join(self.root_poses_train,"automatic", take_uid+".json")
            take_name=self.takes_metadata[take_uid]["take_name"]

            if not os.path.exists(path_camera_json):
                path_camera_json= os.path.join(self.root_camera_val, take_uid+".json")
                path_pose_json = os.path.join(self.root_poses_val,"automatic", take_uid+".json")
            if not os.path.exists(path_camera_json):
                print("pass no pose_json, no cam_json in train/val",take_name)
                continue

            camera_json = json.load(open(path_camera_json))
            pose_json = json.load(open(path_pose_json))

            cseq_name = f"{take_name}_{cam_key}"
            if cseq_name not in self.list_valid_take_name_cam:
                print("pass not in valid list", cseq_name)
                continue

            cam_intr = np.array(camera_json[cam_key]['camera_intrinsics'])
            cam_param = {'focal': np.array([cam_intr[0,0], cam_intr[1,1]], dtype=np.float32),
                        'princpt': np.array([cam_intr[0,2], cam_intr[1,2]], dtype=np.float32)}
            

            # openpose json
            try:
                openpose_json = json.load(open(os.path.join(self.root_openpose, f"{take_name}_{cam_key}.json")))
            except:
                openpose_json = {}

                                    
            T_w2c = np.eye(4)
            T_w2c[:3, :] = np.array(camera_json[cam_key]['camera_extrinsics'])

            cam_param['T_w2c'] = T_w2c

            frame_with_pose = [int(frame_idx) for frame_idx in list(pose_json.keys())]
            frame_with_pose.sort()
            self.logger.info(f"Loading: {take_name}_{cam_key}: from frame #{frame_with_pose[0]} to {frame_with_pose[-1]}")

            for image_id in tqdm(range(frame_with_pose[0], frame_with_pose[-1]+1)):

                image_path = os.path.join(take_name, cam_key, f"frame_{image_id:04d}.jpg")
                
                joints3d_arr = np.zeros((self.joint_set["joint_num"],3))-1.
                joints2d_arr = np.zeros((self.joint_set["joint_num"],2))-1.
                joints_flags = np.zeros((self.joint_set["joint_num"],))
                
                is_first_valid_frame = cseq_name not in seq_names
                has_annotation = str(image_id) in pose_json and len(pose_json[str(image_id)])>0
                if has_annotation:
                    current_anno=pose_json[str(image_id)][0]
                    has_annotation = cam_key in current_anno["annotation2D"]
                
                if not has_annotation:
                    print(f"pass sample, no GT in {take_name}, {cam_key}, {image_id}")
                    continue

                joints3d=current_anno["annotation3D"]
                joints2d=current_anno["annotation2D"][cam_key]



                for jidx, joint_name in enumerate(self.joint_set['original_joints_name']):
                    if joint_name in joints3d:
                        joints_flags[jidx] = 1
                        joints3d_arr[jidx] = [joints3d[joint_name]['x'], joints3d[joint_name]['y'], joints3d[joint_name]['z']]
                        joints2d_arr[jidx] = [joints2d[joint_name]['x'], joints2d[joint_name]['y']]

                sum_valid_joints = joints_flags.sum()
                valid_pelvis = joints_flags[self.r_hip_idx]>1e-4 and joints_flags[self.l_hip_idx]>1e-4
                if sum_valid_joints<10 or not valid_pelvis:
                    print(f"pass sample, not enough GT in {take_name}, {cam_key}, {image_id}, {sum_valid_joints} of 17 valid, pelvis: {valid_pelvis}")
                    continue

                #apply camera extrinsic
                joints4d = np.ones((self.joint_set["joint_num"],4))
                joints4d[:,:3] = joints3d_arr
                
                joints4d_cam = np.dot(T_w2c, joints4d.T).T
                joints3d_arr = joints4d_cam[:,:3]
                joints3d_arr = np.where(joints_flags[:,None]>1e-4, joints3d_arr, -1)


                #add pelvis and neck for GT 3D and 2D
                if joints_flags[self.r_shd_idx]>1e-4 and joints_flags[self.l_shd_idx]>1e-4:
                    joints3d_arr[self.neck_idx] = (joints3d_arr[self.r_shd_idx] + joints3d_arr[self.l_shd_idx])/2
                    joints2d_arr[self.neck_idx] = (joints2d_arr[self.r_shd_idx] + joints2d_arr[self.l_shd_idx])/2
                    joints_flags[self.neck_idx] = 1. 
                if joints_flags[self.r_hip_idx]>1e-4 and joints_flags[self.l_hip_idx]>1e-4:
                    joints3d_arr[self.pelvis_idx] = (joints3d_arr[self.r_hip_idx] + joints3d_arr[self.l_hip_idx])/2
                    joints2d_arr[self.pelvis_idx] = (joints2d_arr[self.r_hip_idx] + joints2d_arr[self.l_hip_idx])/2
                    joints_flags[self.pelvis_idx] = 1.

                joint_img = np.concatenate((joints2d_arr, joints_flags[:,None]), axis=-1).astype(np.float32)
                

                #skip inaccurate intrinsic:
                joints2d_hom = np.dot(cam_intr, joints3d_arr.T).T
                joints2d_proj = joints2d_hom[:,:2]/(joints2d_hom[:,2:3]+1e-4)
                proj_anno_diff = (joints2d_proj-joints2d_arr)[joints_flags==1]
                proj_anno_diff = np.linalg.norm(proj_anno_diff, axis=1)
                
                if np.fabs(proj_anno_diff).mean()>20:
                    print(f"pass sample, suspecting unreliable annotation, {take_name}, {cam_key}, {image_id}, {np.fabs(proj_anno_diff).mean()}, {np.fabs(proj_anno_diff).max()}")
                    continue

                seq_names.append(cseq_name)
                datalist.append({'ann_id': aid,
                                'img_id': image_id,
                                'img_path': image_path,
                                'img_shape': self.image_hw,
                                #'bbox': bbox,
                                'joint_img': joint_img,
                                'joint_world': joints4d[:,:3],
                                'joint_cam': joints3d_arr,
                                'joint_cam_valid': joints_flags,
                                'cam_param': cam_param,
                                'seq_name': cseq_name,})
                aid+=1

        
        #unique_seq_names, seq_start_end_idx = get_video_start_end_indices(seq_names)

        #print(f"Total number of sequences: {len(unique_seq_names)}")
        #print(f"Total number of samples: {len(datalist)}")
        
        return datalist#, unique_seq_names, seq_start_end_idx

    def get_image_from_lmdb(self, image_path):
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





def get_camera_params_and_world_joints_from_egoexo(dataset):
    dict_seq_cam_params = {}
    dict_seq_joints_world = {}
    for data_item in dataset.datalist:
        cseq_tag, ccam_tag = '_'.join(data_item['seq_name'].split('_')[:-1]), data_item['seq_name'].split('_')[-1]
        if cseq_tag not in dict_seq_cam_params:
            dict_seq_cam_params[cseq_tag] = {}
            dict_seq_joints_world[cseq_tag] = {}

        # assign cam_params to each sequence  
        ccam_param = data_item['cam_param']

        focal = ccam_param['focal']  # [fx, fy]
        princpt = ccam_param['princpt']  # [cx, cy]
        T_w2c = ccam_param['T_w2c']  # 4
        # Build camera intrinsics matrix K
        K = np.array([[focal[0], 0, princpt[0]],
                        [0, focal[1], princpt[1]],
                        [0, 0, 1]], dtype=np.float32)
        # Extract rotation and translation from T_w2c
        R_w2c = T_w2c[:3, :3]  # 3x3 rotation matrix
        t_w2c = T_w2c[:3, 3]   # 3x1 translation vector

        # Convert rotation matrix to rotation vector
        rvec, _ = cv2.Rodrigues(R_w2c)
        tvec = t_w2c.reshape(-1, 1)

        dict_seq_cam_params[cseq_tag][ccam_tag] = {
            'K': K,
            'rvec': rvec.flatten(),
            'tvec': tvec.flatten(),
            'focal': focal,
            'princpt': princpt,
            'T_w2c': T_w2c,}

            
        frame_id = data_item['img_id']
        if frame_id not in dict_seq_joints_world[cseq_tag]:
            cjoint_world = data_item['joint_world']
            cjoint_world_flag = data_item['joint_cam_valid'].reshape(-1, 1)
            cjoint_world = np.concatenate([cjoint_world, cjoint_world_flag], axis=-1)  # (17, 4) [x, y, z, valid_flag]
            cjoint_world = transform_joint_to_other_db(cjoint_world, dataset.joint_set['joints_name'], VIT_JOINTS_NAME)
            dict_seq_joints_world[cseq_tag][frame_id] = cjoint_world
    
    return dict_seq_cam_params, dict_seq_joints_world


