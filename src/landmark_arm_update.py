#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, math
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import cv2
import joblib
from collections import deque, defaultdict

# Jetson-friendly: choose device/precision dynamically
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None

from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity

# ===== messages
from landmarks_detection_ROS.msg import landmark, landmark_array

# ==== Config & paths (ARM-safe defaults)
import rospkg

rospack = rospkg.RosPack()
# Prefer local package assets to avoid cross-package path confusion on devices
try:
    package_path = rospack.get_path(rospy.get_param("~ros_package_name", "landmarks_detection_ROS"))


except rospkg.ResourceNotFound:
    package_path = rospack.get_path("landmarks_detection")
# Model/codebook defaults (override via ROS params in launch)
YOLO_MODEL_PATH = rospy.get_param("~yolo_model", f"{package_path}/weights/last.pt")
CODEBOOK_PATH   = rospy.get_param("~codebook",   f"{package_path}/codebook/codebook_kNN.joblib")
IDF_PATH        = rospy.get_param("~idf",        f"{package_path}/codebook/idf.npy")

# Jetson-specific knobs
CUDA_OK         = TORCH_AVAILABLE and torch.cuda.is_available()
DEVICE          = rospy.get_param("~device", 0 if CUDA_OK else 'cpu')
IMG_SIZE        = rospy.get_param("~imgsz", 512)     # smaller default for Nano
FP16            = rospy.get_param("~fp16", bool(CUDA_OK))

CONF_THRES      = rospy.get_param("~conf_thres", 0.5)
SIM_THRESH_BASE = rospy.get_param("~sim_thresh", 0.12)
SNAP_MAX        = rospy.get_param("~snapshots",  7)
USE_DEPTH       = rospy.get_param("~use_depth",  True)
USE_SURF        = rospy.get_param("~use_surf",   False)  # keep SURF off by default on ARM
SIFT_NFEAT      = rospy.get_param("~sift_nfeatures", 800)  # slightly smaller than desktop
DEPTH_TOPIC     = rospy.get_param("~depth_topic","/camera/aligned_depth_to_color/image_raw")
COLOR_TOPIC     = rospy.get_param("~color_topic","/camera/color/image_raw")
INFO_TOPIC      = rospy.get_param("~info_topic", "/camera/color/camera_info")
MAX_DEPTH_M     = rospy.get_param("~max_depth_m", 6.0)
DEPTH_KERNEL    = rospy.get_param("~depth_kernel", 7)
PUBLISH_MARKERS = rospy.get_param("~publish_markers", True)

# ==== re-ID upgrades
USE_DEBOUNCE        = rospy.get_param("~use_debounce", True)
DEBOUNCE_WINDOW     = rospy.get_param("~debounce_window", 5)
CENTER_MATCH_PIX    = rospy.get_param("~center_match_pix", 60)

USE_PROTOTYPE       = rospy.get_param("~use_prototype", True)
PROTOTYPE_MAXLEN    = rospy.get_param("~prototype_maxlen", SNAP_MAX)

USE_PER_CLASS_TH    = rospy.get_param("~use_per_class_th", False)
CLASS_THRESH        = rospy.get_param("~class_thresholds", {})

USE_SPATIAL_SCORE   = rospy.get_param("~use_spatial_score", False)
W_APP               = rospy.get_param("~w_app", 0.55)
W_DIST              = rospy.get_param("~w_dist", 0.25)
W_AGE               = rospy.get_param("~w_age", 0.20)
AGE_TAU_SEC         = rospy.get_param("~age_tau_sec", 5.0)
DIST_CLIP_M         = rospy.get_param("~dist_clip_m", 1.0)

USE_SPACE_GATING    = rospy.get_param("~use_space_gating", False)
GATE_DIST_M         = rospy.get_param("~gate_dist_m", 0.7)

if TORCH_AVAILABLE and CUDA_OK:
    # improves throughput for fixed-size inference on Jetson
    torch.backends.cudnn.benchmark = True

bridge = CvBridge()

# =========================
# BoVW encoder
# =========================
class BoVW:
    def __init__(self, codebook_path, idf_path):
        self.kmeans = joblib.load(codebook_path)
        self.idf = np.load(idf_path).astype(np.float32)
        self.K = int(self.kmeans.n_clusters)
        assert self.idf.shape[0] == self.K, "IDF size != n_clusters"
        # detector: SIFT mặc định, SURF nếu bật (ARM: keep lightweight)
        if USE_SURF and hasattr(cv2, "xfeatures2d"):
            self.det = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
            self.desc_dim = 64
        else:
            # SIFT may not be compiled in some OpenCV builds; try/except
            try:
                self.det = cv2.SIFT_create(nfeatures=int(SIFT_NFEAT))
                self.desc_dim = 128
            except Exception as e:
                rospy.logwarn_throttle(10.0, f"SIFT unavailable on this build: {e}. BoVW may not work with current codebook.")
                # Fallback to ORB to avoid crash; descriptor dims won't match trained codebook
                self.det = cv2.ORB_create(nfeatures=int(max(500, SIFT_NFEAT)))
                self.desc_dim = 32

    def roi_desc(self, gray):
        kps, desc = self.det.detectAndCompute(gray, None)
        return desc

    def tfidf(self, desc):
        if desc is None or len(desc) == 0:
            return None
        labels = self.kmeans.predict(desc.astype(np.float32))
        h, _ = np.histogram(labels, bins=np.arange(self.K+1))
        h = h.astype(np.float32)
        # TF -> IDF -> Hellinger -> L2
        s = h.sum()
        if s > 0:
            h /= s
        h *= self.idf
        h = np.sqrt(np.maximum(h, 0))
        n = np.linalg.norm(h) + 1e-9
        return h / n

# =========================
# Memory with upgrades
# =========================
class Memory:
    def __init__(self, sim_thresh_base=0.12, maxlen=7):
        self.next_id = 0
        self.cls  = {}
        self.vecs = {}
        self.proto = {}
        self.meta = {}
        self.maxlen = maxlen
        self.sim_base = sim_thresh_base

    def _th_for_class(self, cls_name:str)->float:
        if USE_PER_CLASS_TH and isinstance(CLASS_THRESH, dict) and cls_name in CLASS_THRESH:
            return float(CLASS_THRESH[cls_name])
        return float(self.sim_base)

    def _prototype_of(self, lid:int):
        if USE_PROTOTYPE:
            V = list(self.vecs[lid])
            if len(V) == 1:
                p = V[0]
            else:
                p = np.mean(np.stack(V, axis=0), axis=0)
            p = p / (np.linalg.norm(p)+1e-9)
            self.proto[lid] = p
            return p
        else:
            return self.vecs[lid]

    def _cosine(self, a, b):
        return float(cosine_similarity(a.reshape(1,-1), b.reshape(1,-1))[0,0])

    def _spatial_score(self, sim, xyz, lid):
        s_app = max(0.0, min(1.0, sim))
        s_dist = 0.0
        s_age  = 0.0
        now = time.time()
        m = self.meta.get(lid, {})
        if xyz is not None and "last_xyz" in m and m["last_xyz"] is not None:
            dist = float(np.linalg.norm(np.array(xyz) - np.array(m["last_xyz"])))
            dist = min(dist, DIST_CLIP_M)
            s_dist = 1.0 - (dist / DIST_CLIP_M)
        if "last_seen" in m:
            dt = max(0.0, now - m["last_seen"])
            s_age = math.exp(-dt / max(1e-6, AGE_TAU_SEC))
        S = W_APP*s_app + W_DIST*s_dist + W_AGE*s_age
        return S

    def _gate_space(self, xyz, lid)->bool:
        if not USE_SPACE_GATING or xyz is None:
            return True
        m = self.meta.get(lid, {})
        if "last_xyz" not in m or m["last_xyz"] is None:
            return True
        dist = float(np.linalg.norm(np.array(xyz) - np.array(m["last_xyz"])))
        return dist <= GATE_DIST_M

    def best_match(self, cls_name, vec, xyz=None, last_box=None):
        if vec is None:
            return None, 0.0, 0.0
        best_id, best_sim, best_score = None, -1.0, -1.0
        for lid, c in self.cls.items():
            if c != cls_name:
                continue
            if not self._gate_space(xyz, lid):
                continue
            proto = self._prototype_of(lid)
            sim = self._cosine(vec, proto)
            score = self._spatial_score(sim, xyz, lid) if USE_SPATIAL_SCORE else sim
            if score > best_score:
                best_id, best_sim, best_score = lid, sim, score
        return best_id, best_sim, best_score

    def update_or_new(self, cls_name, vec, xyz=None, last_box=None, matched_id=None):
        now = time.time()
        if matched_id is None:
            lid = self.next_id; self.next_id += 1
            self.cls[lid] = cls_name
            if USE_PROTOTYPE:
                self.vecs[lid] = deque([vec], maxlen=max(1, PROTOTYPE_MAXLEN))
            else:
                self.vecs[lid] = vec
            self.meta[lid] = {"last_xyz": xyz, "last_seen": now, "last_box": last_box}
            self.proto[lid] = vec / (np.linalg.norm(vec)+1e-9)
            return lid
        else:
            lid = matched_id
            if USE_PROTOTYPE:
                self.vecs[lid].append(vec)
            else:
                self.vecs[lid] = 0.8*self.vecs[lid] + 0.2*vec
                self.vecs[lid] /= (np.linalg.norm(self.vecs[lid])+1e-9)
            self.meta[lid]["last_xyz"]  = xyz if xyz is not None else self.meta[lid].get("last_xyz")
            self.meta[lid]["last_seen"] = now
            self.meta[lid]["last_box"]  = last_box
            self._prototype_of(lid)
            return lid

    def th_ok(self, cls_name, sim)->bool:
        return sim >= self._th_for_class(cls_name)

# =========================
# Node
# =========================
class Node:
    def __init__(self):
        # Load YOLO with device/precision suitable for Jetson
        self.model = YOLO(YOLO_MODEL_PATH)
        self.bovw  = BoVW(CODEBOOK_PATH, IDF_PATH)
        self.mem   = Memory(SIM_THRESH_BASE, SNAP_MAX)

        self.sub_color = rospy.Subscriber(COLOR_TOPIC, Image, self.cb_color, queue_size=1, buff_size=2**22)
        self.sub_depth = rospy.Subscriber(DEPTH_TOPIC, Image, self.cb_depth, queue_size=1, buff_size=2**22) if USE_DEPTH else None
        self.sub_info  = rospy.Subscriber(INFO_TOPIC, CameraInfo, self.cb_info, queue_size=1)

        self.pub_lm    = rospy.Publisher("landmarks", landmark_array, queue_size=1)
        self.pub_mk    = rospy.Publisher("landmark_markers", MarkerArray, queue_size=1) if PUBLISH_MARKERS else None

        self.depth = None
        self.K = None
        self.frame_id = "camera_color_optical_frame"

        self.prev_centers = []
        self.miss_count   = defaultdict(int)

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

    def _nearest_prev(self, cx, cy):
        if not self.prev_centers:
            return None
        pts = np.array([[px,py] for (px,py,_) in self.prev_centers], dtype=np.float32)
        d = np.linalg.norm(pts - np.array([cx,cy], dtype=np.float32), axis=1)
        j = int(np.argmin(d))
        if d[j] <= CENTER_MATCH_PIX:
            return self.prev_centers[j][2]
        return None

    def cb_color(self, msg):
        im = bridge.imgmsg_to_cv2(msg, "bgr8")
        H,W = im.shape[:2]
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        res = self.model.predict(source=im, conf=CONF_THRES, verbose=False, device=DEVICE, imgsz=int(IMG_SIZE), half=bool(FP16))
        if not res:
            return
        r = res[0]
        if r.boxes is None:
            return

        now_ros = rospy.Time.now()
        now_sec = time.time()

        lms = []
        markers = []
        new_prev = []

        for b in r.boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0].cpu().numpy())
            x1 = max(0,min(W-1,x1)); x2 = max(0,min(W-1,x2))
            y1 = max(0,min(H-1,y1)); y2 = max(0,min(H-1,y2))
            if x2<=x1 or y2<=y1:
                continue

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
                else:
                    cx = (x1+x2)//2; cy = (y1+y2)//2
            else:
                cx = (x1+x2)//2; cy = (y1+y2)//2

            best_id, best_sim, best_score = self.mem.best_match(cls_name, vec, xyz, last_box=[x1,y1,x2,y2])

            assigned_id = None
            pass_th = self.mem.th_ok(cls_name, best_sim)

            prev_id_near = self._nearest_prev(cx, cy) if USE_DEBOUNCE else None

            if best_id is not None and pass_th:
                assigned_id = best_id
                if prev_id_near is not None:
                    self.miss_count[prev_id_near] = 0
            else:
                if USE_DEBOUNCE and prev_id_near is not None and self.miss_count[prev_id_near] < DEBOUNCE_WINDOW:
                    assigned_id = prev_id_near
                    self.miss_count[prev_id_near] += 1
                else:
                    assigned_id = self.mem.update_or_new(cls_name, vec, xyz, [x1,y1,x2,y2], matched_id=None)
                    self.miss_count[assigned_id] = 0

            if assigned_id == best_id and best_id is not None:
                self.mem.update_or_new(cls_name, vec, xyz, [x1,y1,x2,y2], matched_id=best_id)

            new_prev.append((cx, cy, assigned_id))

            lm = landmark()
            lm.id  = int(assigned_id)
            lm.cls = cls_name
            lm.sim = float(best_sim)
            lm.bbox = [x1,y1,x2,y2]
            if xyz is not None:
                lm.x, lm.y, lm.z = [float(v) for v in xyz]
            else:
                lm.x = lm.y = lm.z = float('nan')
            lms.append(lm)

            if PUBLISH_MARKERS:
                m = Marker()
                m.header = Header(frame_id=self.frame_id, stamp=now_ros)
                m.ns = "landmarks"
                m.id = int(assigned_id)
                m.type = Marker.TEXT_VIEW_FACING
                m.action = Marker.ADD
                if xyz is not None:
                    m.pose.position.x, m.pose.position.y, m.pose.position.z = xyz
                else:
                    m.pose.position.z = 1.0
                m.scale.z = 0.08
                m.color.r, m.color.g, m.color.b, m.color.a = (1.0,1.0,0.0,1.0)

                tag = f"ID#{assigned_id} {cls_name} s={best_sim:.2f}"
                if USE_DEBOUNCE and prev_id_near is not None and assigned_id == prev_id_near and (best_id is None or not pass_th):
                    tag += f" pending({self.miss_count[prev_id_near]}/{DEBOUNCE_WINDOW})"
                m.text = tag
                markers.append(m)

            cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(im, f"ID#{assigned_id} {cls_name} s={best_sim:.2f}",
                        (x1, max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        self.prev_centers = new_prev

        arr = landmark_array()
        arr.header = Header(frame_id=self.frame_id, stamp=now_ros)
        arr.landmarks = lms
        self.pub_lm.publish(arr)

        if PUBLISH_MARKERS:
            mk = MarkerArray(markers=markers)
            self.pub_mk.publish(mk)

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