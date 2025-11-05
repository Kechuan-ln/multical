#!/usr/bin/env python3
"""
多相机完整同步工作流程（GoPro + PrimeColor + Mocap）

设计理念：
1. 自动检测目录结构 - 只需提供根目录路径
2. Graceful Degradation - 缺失组件不影响其他步骤
3. 详细报告 - 记录所有检测结果和执行状态

使用示例：
    # 自动检测所有组件
    python sync_multi_camera_workflow.py \\
        --input_dir /path/to/data_root \\
        --output_dir /path/to/output

预期目录结构：
    data_root/
    ├── gopro_raw/ (或 gopro/)        # GoPro视频（必需）
    │   ├── cam01/Video.MP4
    │   ├── cam02/Video.MP4
    │   └── ...
    ├── primecolor_raw/ (或 primecolor/)  # PrimeColor视频（可选）
    │   └── Video.avi
    ├── mocap/                         # Mocap CSV（可选）
    │   └── video.csv (或其他.csv)
    └── qr_sync.mp4 (或 anchor.mp4)   # QR anchor视频（可选）
"""

import os
import sys
import json
import glob
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def print_section(title: str, level: int = 1):
    """打印分隔线"""
    if level == 1:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
    else:
        print("\n" + "-" * 60)
        print(title)
        print("-" * 60)


class DataDetector:
    """自动检测数据目录结构"""

    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
        self.detected = {
            'gopro_dir': None,
            'gopro_count': 0,
            'primecolor_video': None,
            'anchor_video': None,
            'mocap_csv': None
        }

    def detect_gopro(self) -> Optional[str]:
        """检测GoPro目录"""
        # 查找可能的GoPro目录名称
        possible_names = ['gopro_raw', 'gopro', 'GoPro_raw', 'GoPro']

        for name in possible_names:
            gopro_dir = self.input_dir / name
            if gopro_dir.exists() and gopro_dir.is_dir():
                # 检查是否有视频
                videos = list(gopro_dir.glob('*/*.MP4')) + list(gopro_dir.glob('*/*.mp4'))
                if not videos:
                    videos = list(gopro_dir.glob('*.MP4')) + list(gopro_dir.glob('*.mp4'))

                if videos:
                    self.detected['gopro_dir'] = str(gopro_dir)
                    # 统计相机数量
                    camera_dirs = set()
                    for v in videos:
                        parent = v.parent
                        if parent != gopro_dir:
                            camera_dirs.add(parent.name)

                    self.detected['gopro_count'] = len(camera_dirs) if camera_dirs else len(videos)
                    return str(gopro_dir)

        return None

    def detect_primecolor(self) -> Optional[str]:
        """检测PrimeColor视频"""
        # 查找可能的PrimeColor目录或文件
        possible_dirs = ['primecolor_raw', 'primecolor', 'PrimeColor_raw', 'PrimeColor']

        # 方法1: 在特定目录下查找
        for dirname in possible_dirs:
            primecolor_dir = self.input_dir / dirname
            if primecolor_dir.exists() and primecolor_dir.is_dir():
                # 查找视频文件
                videos = list(primecolor_dir.glob('*.avi')) + \
                        list(primecolor_dir.glob('*.AVI')) + \
                        list(primecolor_dir.glob('*.mp4')) + \
                        list(primecolor_dir.glob('*.MP4'))

                if videos:
                    self.detected['primecolor_video'] = str(videos[0])
                    return str(videos[0])

        # 方法2: 在根目录下查找（备用）
        possible_files = ['Video.avi', 'video.avi', 'primecolor.avi', 'sync.avi']
        for filename in possible_files:
            video_path = self.input_dir / filename
            if video_path.exists():
                self.detected['primecolor_video'] = str(video_path)
                return str(video_path)

        return None

    def detect_anchor(self) -> Optional[str]:
        """检测QR anchor视频"""
        possible_names = ['qr_sync.mp4', 'anchor.mp4', 'qr_anchor.mp4', 'QR_sync.mp4']

        for name in possible_names:
            anchor_path = self.input_dir / name
            if anchor_path.exists():
                self.detected['anchor_video'] = str(anchor_path)
                return str(anchor_path)

        return None

    def detect_mocap(self) -> Optional[str]:
        """检测Mocap CSV"""
        mocap_dir = self.input_dir / 'mocap'

        if mocap_dir.exists() and mocap_dir.is_dir():
            # 查找CSV文件
            csv_files = list(mocap_dir.glob('*.csv'))
            if csv_files:
                # 优先选择名字包含'video'的
                for csv in csv_files:
                    if 'video' in csv.name.lower():
                        self.detected['mocap_csv'] = str(csv)
                        return str(csv)

                # 否则选择第一个
                self.detected['mocap_csv'] = str(csv_files[0])
                return str(csv_files[0])

        # 备用：在根目录查找
        csv_files = list(self.input_dir.glob('*.csv'))
        if csv_files:
            self.detected['mocap_csv'] = str(csv_files[0])
            return str(csv_files[0])

        return None

    def detect_all(self) -> Dict:
        """检测所有组件"""
        print_section("步骤0: 自动检测数据结构")
        print(f"扫描目录: {self.input_dir}")

        # 检测各个组件
        self.detect_gopro()
        self.detect_primecolor()
        self.detect_anchor()
        self.detect_mocap()

        # 打印检测结果
        print("\n检测结果:")

        if self.detected['gopro_dir']:
            print(f"  ✅ GoPro: {self.detected['gopro_count']} 个相机")
            print(f"     路径: {self.detected['gopro_dir']}")
        else:
            print(f"  ❌ GoPro: 未找到")

        if self.detected['primecolor_video']:
            print(f"  ✅ PrimeColor: {Path(self.detected['primecolor_video']).name}")
            print(f"     路径: {self.detected['primecolor_video']}")
        else:
            print(f"  ⊘  PrimeColor: 未找到（跳过）")

        if self.detected['anchor_video']:
            print(f"  ✅ QR Anchor: {Path(self.detected['anchor_video']).name}")
            print(f"     路径: {self.detected['anchor_video']}")
        else:
            print(f"  ⊘  QR Anchor: 未找到（跳过QR验证和PrimeColor同步）")

        if self.detected['mocap_csv']:
            print(f"  ✅ Mocap CSV: {Path(self.detected['mocap_csv']).name}")
            print(f"     路径: {self.detected['mocap_csv']}")
        else:
            print(f"  ⊘  Mocap CSV: 未找到（跳过）")

        return self.detected


class SyncWorkflow:
    """多相机同步工作流程"""

    def __init__(self, input_dir: str, output_dir: str, sync_mode: str = 'ultrafast',
                 qr_start_duration: float = 30.0, qr_end_duration: float = 60.0,
                 create_stacked: bool = False, videos_per_row: int = 3):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sync_mode = sync_mode  # fast_copy, ultrafast, accurate
        self.qr_start_duration = qr_start_duration  # 视频开头扫描时长（步骤2、3）
        self.qr_end_duration = qr_end_duration      # 视频结尾扫描时长（步骤4）
        self.create_stacked = create_stacked        # 是否生成堆叠视频
        self.videos_per_row = videos_per_row        # 堆叠视频每行视频数

        # 自动检测数据
        self.detector = DataDetector(self.input_dir)
        self.detected = self.detector.detect_all()

        # 结果跟踪
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'input_dir': str(self.input_dir),
            'detected_components': self.detected,
            'steps': {},
            'errors': [],
            'warnings': []
        }

    def add_result(self, step_name: str, status: str, data: Dict = None, message: str = None):
        """记录步骤结果"""
        self.results['steps'][step_name] = {
            'status': status,  # success / skipped / failed
            'data': data or {},
            'message': message or ''
        }

    def add_error(self, message: str):
        """记录错误"""
        self.results['errors'].append(message)
        print(f"  ❌ 错误: {message}")

    def add_warning(self, message: str):
        """记录警告"""
        self.results['warnings'].append(message)
        print(f"  ⚠️  警告: {message}")

    def step1_gopro_timecode_sync(self) -> bool:
        """步骤1: GoPro官方timecode同步"""
        print_section("步骤1: GoPro官方timecode同步")

        if not self.detected['gopro_dir']:
            self.add_result('gopro_timecode_sync', 'skipped', message='未检测到GoPro视频')
            print("  ⊘ 跳过: 未检测到GoPro视频")
            return False

        print(f"  GoPro相机数: {self.detected['gopro_count']}")

        # 执行同步
        gopro_synced_dir = self.output_dir / 'gopro_synced'

        cmd = [
            sys.executable,
            'scripts/sync_timecode.py',
            '--src_tag', self.detected['gopro_dir'],
            '--out_tag', str(gopro_synced_dir)
        ]

        # 根据sync_mode添加参数
        if self.sync_mode == 'fast_copy':
            cmd.append('--fast_copy')
            print("  模式: fast_copy（最快，关键帧精度，可能有0-2秒误差）")
        elif self.sync_mode == 'ultrafast':
            print("  模式: ultrafast（快速且帧精确，推荐）")
            # 默认模式，不需要额外参数
        elif self.sync_mode == 'accurate':
            cmd.append('--accurate')
            print("  模式: accurate（最慢但最精确，medium preset）")

        print(f"  执行同步...")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # 读取meta_info.json
            meta_info_path = gopro_synced_dir / 'meta_info.json'
            if meta_info_path.exists():
                with open(meta_info_path, 'r') as f:
                    meta_info = json.load(f)

                self.results['gopro_meta_info'] = meta_info
                num_cameras = len(meta_info.get('info_cam', {}))

                self.add_result('gopro_timecode_sync', 'success', {
                    'num_cameras': num_cameras,
                    'output_dir': str(gopro_synced_dir),
                    'meta_info_path': str(meta_info_path)
                })

                print(f"  ✅ 成功同步 {num_cameras} 个GoPro相机")
                return True
            else:
                self.add_warning("meta_info.json未生成")
                return False
        else:
            self.add_error(f"GoPro同步失败: {result.stderr[:200]}")
            self.add_result('gopro_timecode_sync', 'failed', message=result.stderr[:200])
            return False

    def step2_gopro_qr_verification(self) -> Optional[Dict]:
        """步骤2: GoPro同步质量验证（QR码）"""
        print_section("步骤2: GoPro同步质量验证（QR码）")

        # 检查前置条件
        if 'gopro_timecode_sync' not in self.results['steps'] or \
           self.results['steps']['gopro_timecode_sync']['status'] != 'success':
            print("  ⊘ 跳过: GoPro同步未完成")
            self.add_result('gopro_qr_verification', 'skipped', message='GoPro同步未完成')
            return None

        if not self.detected['anchor_video']:
            print("  ⊘ 跳过: 未检测到QR anchor视频")
            self.add_result('gopro_qr_verification', 'skipped', message='未检测到QR anchor视频')
            return None

        gopro_synced_dir = self.output_dir / 'gopro_synced'
        verification_json = self.output_dir / 'gopro_qr_verification.json'

        cmd = [
            sys.executable,
            'sync/verify_gopro_sync_with_qr.py',
            '--gopro_dir', str(gopro_synced_dir),
            '--anchor_video', self.detected['anchor_video'],
            '--start_duration', str(self.qr_start_duration),
            '--end_duration', '0',  # 步骤2只扫描开头
            '--save_json', str(verification_json)
        ]

        print(f"  执行QR验证...")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and verification_json.exists():
            with open(verification_json, 'r') as f:
                verification = json.load(f)

            sync_quality = verification.get('verification', {}).get('sync_quality', 'unknown')
            verified_cameras = verification.get('verification', {}).get('verified_cameras', [])
            failed_cameras = verification.get('verification', {}).get('failed_cameras', [])
            unverified_cameras = verification.get('verification', {}).get('unverified_cameras', [])

            self.add_result('gopro_qr_verification', 'success', {
                'sync_quality': sync_quality,
                'verified_cameras': verified_cameras,
                'failed_cameras': failed_cameras,
                'unverified_cameras': unverified_cameras,
                'verification_path': str(verification_json)
            })

            # 显示验证结果
            print(f"  同步质量: {sync_quality}")
            print(f"  ✅ 验证成功: {len(verified_cameras)} 个相机 - {', '.join(verified_cameras) if verified_cameras else '无'}")
            if failed_cameras:
                print(f"  ❌ 验证失败: {len(failed_cameras)} 个相机 - {', '.join(failed_cameras)}")
                self.add_warning(f"{len(failed_cameras)}个相机验证失败（同步偏移过大）")
            print(f"  ⊘  未验证: {len(unverified_cameras)} 个相机 - {', '.join(unverified_cameras) if unverified_cameras else '无'}")

            if sync_quality == 'poor':
                self.add_warning(f"同步质量较差: {sync_quality}")

            return verification
        else:
            self.add_warning(f"QR验证失败: {result.stderr[:200]}")
            self.add_result('gopro_qr_verification', 'failed', message=result.stderr[:200])
            return None

    def step3_primecolor_sync(self) -> bool:
        """步骤3: PrimeColor同步"""
        print_section("步骤3: PrimeColor与GoPro同步")

        # 检查前置条件
        if not self.detected['primecolor_video']:
            print("  ⊘ 跳过: 未检测到PrimeColor视频")
            self.add_result('primecolor_sync', 'skipped', message='未检测到PrimeColor视频')
            return False

        if not self.detected['anchor_video']:
            print("  ⊘ 跳过: 未检测到QR anchor视频（PrimeColor同步需要）")
            self.add_result('primecolor_sync', 'skipped', message='未检测到QR anchor视频')
            return False

        # 使用**同步后**的GoPro视频作为参考
        # 这样计算的offset直接适用于PrimeColor（相对于同步后的GoPro）
        gopro_synced_dir = self.output_dir / 'gopro_synced'

        # 优先选择有QR码的相机（从步骤2的验证结果中获取）
        preferred_camera = None
        if 'gopro_qr_verification' in self.results['steps']:
            verification_data = self.results['steps']['gopro_qr_verification'].get('data', {})
            verification_path = verification_data.get('verification_path')
            if verification_path and Path(verification_path).exists():
                with open(verification_path, 'r') as f:
                    verification = json.load(f)
                cameras_with_qr = verification.get('verification', {}).get('cameras_with_qr', [])
                if cameras_with_qr:
                    preferred_camera = cameras_with_qr[0]  # 使用第一个有QR码的相机
                    print(f"  从QR验证结果中选择相机: {preferred_camera}（有QR码）")

        # 找同步后的GoPro视频
        gopro_ref_videos = sorted(gopro_synced_dir.glob('*/Video.MP4'))
        if not gopro_ref_videos:
            gopro_ref_videos = sorted(gopro_synced_dir.glob('*.MP4'))
        if not gopro_ref_videos:
            gopro_ref_videos = sorted(gopro_synced_dir.glob('*/*.mp4'))
        if not gopro_ref_videos:
            gopro_ref_videos = sorted(gopro_synced_dir.glob('*.mp4'))

        if not gopro_ref_videos:
            self.add_warning("未找到同步后的GoPro视频作为参考")
            self.add_result('primecolor_sync', 'skipped', message='未找到GoPro参考视频')
            return False

        # 如果有优先相机，尝试找到它
        gopro_ref_video = None
        if preferred_camera:
            for video in gopro_ref_videos:
                parent = video.parent
                if parent.name == preferred_camera or video.stem == preferred_camera:
                    gopro_ref_video = str(video)
                    break

        # 如果没找到优先相机，使用第一个
        if not gopro_ref_video:
            gopro_ref_video = str(gopro_ref_videos[0])
            if preferred_camera:
                print(f"  ⚠️  未找到{preferred_camera}，使用默认相机")

        ref_name = Path(gopro_ref_video).parent.name if Path(gopro_ref_video).parent != gopro_synced_dir else Path(gopro_ref_video).name
        print(f"  使用GoPro参考: {ref_name} (同步后视频)")

        primecolor_output_dir = self.output_dir / 'primecolor_mocap_synced'

        cmd = [
            sys.executable,
            'sync/sync_primecolor_gopro.py',
            '--gopro_video', gopro_ref_video,
            '--primecolor_video', self.detected['primecolor_video'],
            '--anchor_video', self.detected['anchor_video'],
            '--output_dir', str(primecolor_output_dir),
            '--scan_duration', str(self.qr_start_duration)
        ]

        # 如果有Mocap CSV，添加参数
        if self.detected['mocap_csv']:
            cmd.extend(['--mocap_csv', self.detected['mocap_csv']])
            print(f"  包含Mocap CSV同步")

        print(f"  执行同步...")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # 读取映射参数
            mapping_json = primecolor_output_dir / 'sync_mapping.json'
            if mapping_json.exists():
                with open(mapping_json, 'r') as f:
                    mapping = json.load(f)

                self.add_result('primecolor_sync', 'success', {
                    'output_dir': str(primecolor_output_dir),
                    'mapping': mapping
                })

                print(f"  ✅ PrimeColor同步成功")
                print(f"     偏移: {mapping['offset']:.3f}s ({mapping['offset_frames_primecolor']} 帧)")
                print(f"     FPS比例: {mapping['fps_ratio']:.6f}")

                # 检查是否有Mocap
                mocap_synced = primecolor_output_dir / 'mocap_synced.csv'
                if mocap_synced.exists():
                    self.results['steps']['primecolor_sync']['data']['mocap_synced'] = str(mocap_synced)
                    print(f"     Mocap CSV已同步")

                return True
            else:
                self.add_warning("映射参数JSON未生成")
                return False
        else:
            # 解析错误信息
            stderr = result.stderr

            if 'QR码' in stderr or '共同' in stderr:
                self.add_warning(f"PrimeColor同步失败（QR码检测不足）")
                self.add_result('primecolor_sync', 'failed', message='QR码检测不足')
            else:
                self.add_error(f"PrimeColor同步失败: {stderr[:300]}")
                self.add_result('primecolor_sync', 'failed', message=stderr[:300])

            return False

    def step4_final_verification(self) -> bool:
        """步骤4: 最终验证（所有相机，使用视频结尾）"""
        print_section("步骤4: 最终同步验证（视频结尾）")

        # 检查前置条件
        if not self.detected['anchor_video']:
            print("  ⊘ 跳过: 未检测到QR anchor视频")
            self.add_result('final_verification', 'skipped', message='未检测到QR anchor视频')
            return False

        gopro_synced_dir = self.output_dir / 'gopro_synced'
        if not gopro_synced_dir.exists():
            print("  ⊘ 跳过: GoPro同步未完成")
            self.add_result('final_verification', 'skipped', message='GoPro同步未完成')
            return False

        final_verification_json = self.output_dir / 'final_verification.json'

        cmd = [
            sys.executable,
            'sync/verify_final_sync_all_cameras.py',
            '--gopro_dir', str(gopro_synced_dir),
            '--anchor_video', self.detected['anchor_video'],
            '--end_duration', str(self.qr_end_duration),
            '--save_json', str(final_verification_json)
        ]

        # 如果有同步后的PrimeColor，添加参数
        primecolor_synced = self.output_dir / 'primecolor_mocap_synced' / 'primecolor_synced.mp4'
        if primecolor_synced.exists():
            cmd.extend(['--primecolor_video', str(primecolor_synced)])
            print(f"  包含PrimeColor验证")

        print(f"  扫描结尾 {self.qr_end_duration:.0f}秒...")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and final_verification_json.exists():
            with open(final_verification_json, 'r') as f:
                verification = json.load(f)

            sync_quality = verification.get('verification', {}).get('sync_quality', 'unknown')
            verified_cameras = verification.get('verification', {}).get('verified_cameras', [])
            failed_cameras = verification.get('verification', {}).get('failed_cameras', [])
            unverified_cameras = verification.get('verification', {}).get('unverified_cameras', [])

            self.add_result('final_verification', 'success', {
                'sync_quality': sync_quality,
                'verified_cameras': verified_cameras,
                'failed_cameras': failed_cameras,
                'unverified_cameras': unverified_cameras,
                'verification_path': str(final_verification_json)
            })

            # 显示验证结果
            print(f"  同步质量: {sync_quality}")
            print(f"  ✅ 验证成功: {len(verified_cameras)} 个相机 - {', '.join(verified_cameras) if verified_cameras else '无'}")
            if failed_cameras:
                print(f"  ❌ 验证失败: {len(failed_cameras)} 个相机 - {', '.join(failed_cameras)}")
                self.add_warning(f"{len(failed_cameras)}个相机最终验证失败（同步偏移过大）")
            print(f"  ⊘  未验证: {len(unverified_cameras)} 个相机 - {', '.join(unverified_cameras) if unverified_cameras else '无'}")

            if sync_quality == 'poor':
                self.add_warning(f"最终验证质量较差: {sync_quality}")

            return True
        else:
            self.add_warning(f"最终验证失败: {result.stderr[:200]}")
            self.add_result('final_verification', 'failed', message=result.stderr[:200])
            return False

    def step5_create_stacked_video(self) -> bool:
        """步骤5: 生成堆叠视频（GoPro + PrimeColor）"""
        if not self.create_stacked:
            return True  # 不是错误，只是跳过

        print_section("步骤5: 生成堆叠视频")

        gopro_synced_dir = self.output_dir / 'gopro_synced'
        if not gopro_synced_dir.exists():
            print("  ⊘ 跳过: GoPro同步未完成")
            self.add_result('create_stacked_video', 'skipped', message='GoPro同步未完成')
            return False

        stacked_output = self.output_dir / 'stacked_all_cameras.mp4'

        cmd = [
            sys.executable,
            'sync/create_stacked_video.py',
            '--gopro_dir', str(gopro_synced_dir),
            '--output', str(stacked_output),
            '--layout', 'grid',
            '--videos_per_row', str(self.videos_per_row)
        ]

        # 如果有同步后的PrimeColor，添加参数
        primecolor_synced = self.output_dir / 'primecolor_mocap_synced' / 'primecolor_synced.mp4'
        if primecolor_synced.exists():
            cmd.extend(['--primecolor_video', str(primecolor_synced)])
            print(f"  包含PrimeColor视频")

        print(f"  布局: 网格（每行{self.videos_per_row}个视频）")
        print(f"  生成中...")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and stacked_output.exists():
            file_size_mb = stacked_output.stat().st_size / 1024 / 1024

            self.add_result('create_stacked_video', 'success', {
                'output_path': str(stacked_output),
                'file_size_mb': file_size_mb
            })

            print(f"  ✅ 堆叠视频已生成 ({file_size_mb:.2f} MB)")
            return True
        else:
            self.add_warning(f"堆叠视频生成失败: {result.stderr[:200]}")
            self.add_result('create_stacked_video', 'failed', message=result.stderr[:200])
            return False

    def generate_final_report(self):
        """生成最终报告"""
        print_section("最终报告", level=1)

        # 统计
        total_steps = len(self.results['steps'])
        success_steps = sum(1 for s in self.results['steps'].values() if s['status'] == 'success')
        skipped_steps = sum(1 for s in self.results['steps'].values() if s['status'] == 'skipped')
        failed_steps = sum(1 for s in self.results['steps'].values() if s['status'] == 'failed')

        print(f"\n步骤执行情况:")
        print(f"  总步骤数: {total_steps}")
        print(f"  ✅ 成功: {success_steps}")
        print(f"  ⊘  跳过: {skipped_steps}")
        print(f"  ❌ 失败: {failed_steps}")

        print(f"\n详细结果:")
        for step_name, step_result in self.results['steps'].items():
            status_icon = {
                'success': '✅',
                'skipped': '⊘ ',
                'failed': '❌'
            }.get(step_result['status'], '?')

            print(f"  {status_icon} {step_name}: {step_result['status']}")
            if step_result['message']:
                print(f"     {step_result['message']}")

        if self.results['warnings']:
            print(f"\n警告 ({len(self.results['warnings'])} 个):")
            for warning in self.results['warnings']:
                print(f"  ⚠️  {warning}")

        if self.results['errors']:
            print(f"\n错误 ({len(self.results['errors'])} 个):")
            for error in self.results['errors']:
                print(f"  ❌ {error}")

        # 保存完整报告
        report_path = self.output_dir / 'sync_workflow_report.json'
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"\n📄 完整报告已保存: {report_path}")

        # 输出文件位置
        print(f"\n📁 输出文件:")
        print(f"  根目录: {self.output_dir}")

        if self.results['steps'].get('gopro_timecode_sync', {}).get('status') == 'success':
            print(f"  ├── gopro_synced/          ({self.detected['gopro_count']} 个相机)")

        if self.results['steps'].get('gopro_qr_verification', {}).get('status') == 'success':
            print(f"  ├── gopro_qr_verification.json")

        if self.results['steps'].get('primecolor_sync', {}).get('status') == 'success':
            print(f"  ├── primecolor_mocap_synced/")
            print(f"  │   ├── primecolor_synced.mp4")
            print(f"  │   ├── sync_mapping.json")
            if self.results['steps']['primecolor_sync']['data'].get('mocap_synced'):
                print(f"  │   └── mocap_synced.csv")

        if self.results['steps'].get('final_verification', {}).get('status') == 'success':
            print(f"  ├── final_verification.json")

        if self.results['steps'].get('create_stacked_video', {}).get('status') == 'success':
            print(f"  ├── stacked_all_cameras.mp4  (堆叠视频: GoPro + PrimeColor)")

        print(f"  └── sync_workflow_report.json")

    def run(self):
        """执行完整工作流程"""
        print_section("多相机同步工作流程", level=1)
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")

        # 执行各个步骤
        self.step1_gopro_timecode_sync()
        self.step2_gopro_qr_verification()
        self.step3_primecolor_sync()
        self.step4_final_verification()
        self.step5_create_stacked_video()

        # 生成最终报告
        self.generate_final_report()

        # 返回状态码
        has_errors = len(self.results['errors']) > 0
        has_critical_failures = any(
            step['status'] == 'failed' and step_name in ['gopro_timecode_sync']
            for step_name, step in self.results['steps'].items()
        )

        if has_critical_failures:
            print("\n❌ 工作流程失败（关键步骤失败）")
            return 1
        elif has_errors:
            print("\n⚠️  工作流程完成，但有错误")
            return 2
        else:
            print("\n✅ 工作流程成功完成！")
            return 0


def main():
    parser = argparse.ArgumentParser(
        description='多相机完整同步工作流程（自动检测数据结构）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

    # 自动检测所有组件（推荐）
    python sync_multi_camera_workflow.py \\
        --input_dir /Volumes/FastACIS/GoPro/test_sync \\
        --output_dir /Volumes/FastACIS/GoPro/test_sync/output

    # 使用不同同步模式
    python sync_multi_camera_workflow.py \\
        --input_dir /Volumes/FastACIS/GoPro/test_sync \\
        --output_dir /Volumes/FastACIS/GoPro/test_sync/output \\
        --sync_mode ultrafast  # fast_copy, ultrafast, accurate

    # 生成堆叠视频
    python sync_multi_camera_workflow.py \\
        --input_dir /Volumes/FastACIS/GoPro/test_sync \\
        --output_dir /Volumes/FastACIS/GoPro/test_sync/output \\
        --stacked

预期输入目录结构:

    input_dir/
    ├── gopro_raw/ (或 gopro/)        # GoPro视频（必需）
    │   ├── cam01/Video.MP4
    │   ├── cam02/Video.MP4
    │   └── ...
    ├── primecolor_raw/ (或 primecolor/)  # PrimeColor视频（可选）
    │   └── Video.avi
    ├── mocap/                         # Mocap CSV（可选）
    │   └── video.csv
    └── qr_sync.mp4 (或 anchor.mp4)   # QR anchor视频（可选）

输出目录结构:

    output_dir/
    ├── gopro_synced/                  # 同步后的GoPro视频
    ├── gopro_qr_verification.json     # GoPro同步验证（视频开头）
    ├── primecolor_mocap_synced/       # PrimeColor和Mocap（如果有）
    ├── final_verification.json        # 最终验证（所有相机，视频结尾）
    └── sync_workflow_report.json      # 完整报告
        """
    )

    # 必需参数
    parser.add_argument('--input_dir', required=True,
                       help='输入数据根目录（自动检测内部结构）')
    parser.add_argument('--output_dir', required=True,
                       help='输出目录（所有结果保存在此）')

    # 可选参数
    parser.add_argument('--sync_mode', type=str, default='ultrafast',
                       choices=['fast_copy', 'ultrafast', 'accurate'],
                       help='''视频同步模式:
                           fast_copy - 最快（~1分钟），关键帧精度（可能有0-2秒误差）
                           ultrafast - 快速且帧精确（~5-10分钟，推荐，默认）
                           accurate  - 最慢但最精确（~60分钟，medium preset）''')
    parser.add_argument('--qr_start_duration', type=float, default=30.0,
                       help='视频开头QR码扫描时长（秒），用于步骤2和3，默认30')
    parser.add_argument('--qr_end_duration', type=float, default=60.0,
                       help='视频结尾QR码扫描时长（秒），用于步骤4最终验证，默认60')
    parser.add_argument('--stacked', action='store_true',
                       help='生成堆叠视频（GoPro + PrimeColor，网格布局）')
    parser.add_argument('--videos_per_row', type=int, default=3,
                       help='堆叠视频每行视频数（默认3）')

    args = parser.parse_args()

    # 执行工作流程
    workflow = SyncWorkflow(
        args.input_dir,
        args.output_dir,
        sync_mode=args.sync_mode,
        qr_start_duration=args.qr_start_duration,
        qr_end_duration=args.qr_end_duration,
        create_stacked=args.stacked,
        videos_per_row=args.videos_per_row
    )
    return workflow.run()


if __name__ == '__main__':
    sys.exit(main())
