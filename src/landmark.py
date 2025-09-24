#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, pickle, math
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import cv2
import joblib

from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans

from landmarks_detection_ROS.msg import landmark, landmark_array

# ==== Config ====
import rospkg

rospack = rospkg.RosPack()
package_path = rospack.get_path('landmarks_detection')

YOLO_MODEL_PATH = rospy.get_param("~yolo_model", f"{package_path}/weights/last.pt")
CODEBOOK_PATH   = rospy.get_param("~codebook",   f"{package_path}/codebook/codebook_kNN.joblib")
IDF_PATH        = rospy.get_param("~idf",        f"{package_path}/codebook/idf.npy")
CONF_THRES      = rospy.get_param("~conf_thres", 0.5)
SIM_THRESH      = rospy.get_param("~sim_thresh", 0.12)
SNAP_MAX        = rospy.get_param("~snapshots",  7)
USE_DEPTH       = rospy.get_param("~use_depth",  True)
USE_SURF       = rospy.get_param("~use_surf",   False)
DEPTH_TOPIC     = rospy.get_param("~depth_topic","/camera/aligned_depth_to_color/image_raw")
COLOR_TOPIC     = rospy.get_param("~color_topic","/camera/color/image_raw")
INFO_TOPIC      = rospy.get_param("~info_topic", "/camera/color/camera_info")
MAX_DEPTH_M     = rospy.get_param("~max_depth_m", 6.0)
DEPTH_KERNEL    = rospy.get_param("~depth_kernel", 7)
PUBLISH_MARKERS = rospy.get_param("~publish_markers", True)

# ==== Helpers ====
bridge = CvBridge()

class BoVW:
    def __init__(self, codebook_path, idf_path):
        # load KMeans & IDF
        self.kmeans = joblib.load(codebook_path)
        self.idf = np.load(idf_path).astype(np.float32)  # shape (K,)
        self.K = self.kmeans.n_clusters
        assert self.idf.shape[0] == self.K, "IDF size != n_clusters"
        # SIFT (mặc định) hoặc SURF nếu có contrib và ~use_surf=True
        if USE_SURF and hasattr(cv2, "xfeatures2d"):
            self.det = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        else:
            self.det = cv2.SIFT_create(nfeatures=1200)

    def roi_desc(self, gray):
        kps, desc = self.det.detectAndCompute(gray, None)
        return desc

    def tfidf(self, desc):
        # Fallback TEST-style: nếu không có descriptor => trả về None để cấp ID mới
        if desc is None or len(desc) == 0:
            return None
        labels = self.kmeans.predict(desc.astype(np.float32))
        h, _ = np.histogram(labels, bins=np.arange(self.K+1))
        h = h.astype(np.float32)
        # TF normalize
        s = h.sum()
        if s > 0:
            h = h / s
        # TF-IDF
        h *= self.idf
        # Hellinger + L2
        h = np.sqrt(np.maximum(h, 0))
        n = np.linalg.norm(h) + 1e-9
        return h / n

class Memory:
    def __init__(self, sim_thresh=0.42, snap_max=7):
        self.vecs = []     # list of snapshots (np.array)
        self.cls  = []     # class id/name
        self.ids  = []     # landmark ids
        self.meta = []     # dict per id (store xyz, etc)
        self.next_id = 0
        self.sim_thresh = sim_thresh
        self.snap_max = snap_max

    def match(self, cls_name, vec, xyz=None):
        # Fallback TEST-style: nếu vec is None => cấp ID mới ngay
        if vec is None:
            lid = self.next_id; self.next_id += 1
            self.ids.append(lid)
            self.cls.append(cls_name)
            # Lưu placeholder để giữ vị trí; sẽ được cập nhật khi có vec thực
            self.vecs.append(np.zeros(1, dtype=np.float32))
            self.meta.append({"last_xyz": xyz})
            return lid, 0.0

        best_i, best_sim = -1, -1.0
        for i, v in enumerate(self.vecs):
            if self.cls[i] != cls_name:
                continue
            # Bỏ qua placeholder không khớp kích thước
            if v.ndim==1 and v.size!=vec.size:
                continue
            sim = float(cosine_similarity(vec.reshape(1,-1), v.reshape(1,-1))[0,0])
            if sim > best_sim:
                best_sim, best_i = sim, i
        if best_sim >= self.sim_thresh and best_i >= 0:
            lid = self.ids[best_i]
            # EMA update snapshot (0.9 cũ + 0.1 mới)
            self.vecs[best_i] = 0.9*self.vecs[best_i] + 0.1*vec
            self.vecs[best_i] = self.vecs[best_i] / (np.linalg.norm(self.vecs[best_i]) + 1e-9)
            if xyz is not None:
                self.meta[best_i]["last_xyz"] = xyz
            return lid, best_sim
        # new id
        lid = self.next_id; self.next_id += 1
        self.ids.append(lid)
        self.cls.append(cls_name)
        self.vecs.append(vec)
        self.meta.append({"last_xyz": xyz})
        return lid, best_sim

class Node:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL_PATH)
        self.bovw  = BoVW(CODEBOOK_PATH, IDF_PATH)
        self.mem   = Memory(SIM_THRESH, SNAP_MAX)
        self.sub_color = rospy.Subscriber(COLOR_TOPIC, Image, self.cb_color, queue_size=1, buff_size=2**22)
        self.sub_depth = rospy.Subscriber(DEPTH_TOPIC, Image, self.cb_depth, queue_size=1, buff_size=2**22) if USE_DEPTH else None
        self.sub_info  = rospy.Subscriber(INFO_TOPIC, CameraInfo, self.cb_info, queue_size=1)
        self.pub_lm    = rospy.Publisher("landmarks", landmark_array, queue_size=1)
        self.pub_mk    = rospy.Publisher("landmark_markers", MarkerArray, queue_size=1) if PUBLISH_MARKERS else None
        self.depth = None
        self.K = None  # intrinsics
        self.frame_id = "camera_color_optical_frame"

    def cb_info(self, msg):
        self.frame_id = msg.header.frame_id or self.frame_id
        self.K = (msg.K[0], msg.K[4], msg.K[2], msg.K[5])  # fx, fy, cx, cy

    def cb_depth(self, msg):
        d = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if d.dtype != np.float32:
            d = d.astype(np.float32) / (1000.0 if d.dtype==np.uint16 else 1.0)
        self.depth = d

    def depth_at_bbox(self, bbox):
        if self.depth is None: return None
        x1,y1,x2,y2 = bbox
        cx, cy = (x1+x2)//2, (y1+y2)//2
        h,w = self.depth.shape
        rx1, ry1 = max(0, cx-DEPTH_KERNEL//2), max(0, cy-DEPTH_KERNEL//2)
        rx2, ry2 = min(w, cx+DEPTH_KERNEL//2+1), min(h, cy+DEPTH_KERNEL//2+1)
        patch = self.depth[ry1:ry2, rx1:rx2]
        if patch.size == 0: return None
        z = np.nanmedian(np.where((patch>0)&np.isfinite(patch), patch, np.nan))
        if not np.isfinite(z) or z<=0 or z>MAX_DEPTH_M: return None
        return float(z), int(cx), int(cy)

    def deproject(self, u,v,z):
        if self.K is None: return (0.0,0.0,float(z))
        fx, fy, cx, cy = self.K
        X = (u - cx)/fx * z
        Y = (v - cy)/fy * z
        return float(X), float(Y), float(z)

    def cb_color(self, msg):
        im = bridge.imgmsg_to_cv2(msg, "bgr8")
        H,W = im.shape[:2]
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # YOLO
        res = self.model.predict(source=im, conf=CONF_THRES, verbose=False)
        if not res: return
        r = res[0]
        if r.boxes is None: return

        lms = []
        markers = []
        now = rospy.Time.now()

        for b in r.boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0].cpu().numpy())
            x1 = max(0,min(W-1,x1)); x2 = max(0,min(W-1,x2))
            y1 = max(0,min(H-1,y1)); y2 = max(0,min(H-1,y2))
            if x2<=x1 or y2<=y1: continue

            cls_id = int(b.cls[0].item())
            cls_name = r.names.get(cls_id, str(cls_id))
            roi = gray[y1:y2, x1:x2]

            desc = self.bovw.roi_desc(roi)
            vec  = self.bovw.tfidf(desc)

            xyz = None
            if USE_DEPTH:
                dz = self.depth_at_bbox((x1,y1,x2,y2))
                if dz is not None:
                    z, cx, cy = dz
                    xyz = self.deproject(cx,cy,z)

            lm_id, sim = self.mem.match(cls_name, vec, xyz)

            # message
            lm = landmark()
            lm.id = lm_id
            lm.cls = cls_name
            lm.sim = float(sim)
            lm.bbox = [x1,y1,x2,y2]
            if xyz is not None:
                lm.x, lm.y, lm.z = xyz
            else:
                lm.x = lm.y = lm.z = float('nan')
            lms.append(lm)

            # marker (optional)
            if self.pub_mk:
                m = Marker()
                m.header = Header(frame_id=self.frame_id, stamp=now)
                m.ns = "landmarks"
                m.id = lm_id
                m.type = Marker.TEXT_VIEW_FACING
                m.action = Marker.ADD
                # đặt vị trí gần camera nếu không có depth
                if xyz is not None:
                    m.pose.position.x, m.pose.position.y, m.pose.position.z = xyz
                else:
                    m.pose.position.z = 1.0
                m.scale.z = 0.08
                m.color.r, m.color.g, m.color.b, m.color.a = (1.0,1.0,0.0,1.0)
                m.text = f"ID#{lm_id} {cls_name} s={sim:.2f}"
                markers.append(m)

            # vẽ nhanh cho debug (rqt_image_view sẽ thấy)
            cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(im,f"ID#{lm_id} {cls_name} s={sim:.2f}",(x1,max(15,y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

        # Publish landmarks
        arr = landmark_array()
        arr.header = Header(frame_id=self.frame_id, stamp=now)
        arr.landmarks = lms
        self.pub_lm.publish(arr)

        if self.pub_mk:
            mk = MarkerArray(markers=markers)
            self.pub_mk.publish(mk)

        # (tuỳ chọn) hiển thị debug
        if rospy.get_param("~show_debug", False):
            cv2.imshow("landmarks_detection", im)
            cv2.waitKey(1)

def main():
    rospy.init_node("landmarks_detection_node")
    n = Node()
    rospy.loginfo("landmarks_detection_node started.")
    rospy.spin()

if __name__ == "__main__":
    main()

