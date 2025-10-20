#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create BoVW codebook (MiniBatchKMeans) + IDF cho landmarks (Jetson-friendly).
- Cấu hình ngay bên dưới (không cần tham số dòng lệnh).
- Giảm RAM bằng MiniBatchKMeans + resize ảnh + sample descriptor.
- Mặc định dùng SIFT (128-D). Không fallback sang ORB để tránh sai chiều.
- Xuất:
    - <OUT_PREFIX>_centers.npy   (K x 128, trung lập, nạp ở mọi máy)
    - <OUT_PREFIX>_idf.npy       (K,)
    - <OUT_PREFIX>_meta.json     (metadata K, dim, …)
    - (tuỳ chọn) <OUT_PREFIX>_codebook.joblib  (nếu muốn dùng sklearn cùng phiên bản)

Chạy:
    python3 create_codebook_arm.py
"""

import os, glob, json
import numpy as np
import cv2
from tqdm import tqdm

# ===========================
#           CẤU HÌNH
# ===========================
# Thư mục ảnh (glob). VD: "../dataset/train/images/*.jpg"
IMG_GLOB = "/home/quangtrung/landmarks_ws/src/landmarks_detection_ROS/dataset/train/images/*.jpg"

# Số cụm (visual words). 512–4096 tuỳ dữ liệu/khả năng.
K = 4096

# Tổng descriptor tối đa để train KMeans (giảm => nhẹ RAM, nhanh hơn).
MAX_TOTAL = 2_000_000

# Số descriptor tối đa lấy từ mỗi ảnh.
MAX_FEATS_PER_IMG = 800

# Resize ảnh về sao cho max(h, w) = RESIZE_MAX (0 để tắt).
RESIZE_MAX = 640

# Batch size cho MiniBatchKMeans. Nano thường 4k–16k.
BATCH_SIZE = 8192

# Số lần khởi tạo KMeans (KHÔNG dùng "auto" trên sklearn cũ).
N_INIT = 3

# Seed cho reproducibility
SEED = 0

# Prefix tên file output
OUT_PREFIX = "codebook"

# Xuất thêm file .joblib (chỉ nên bật nếu train + infer cùng phiên bản sklearn)
SAVE_JOBLIB = False

# Tính IDF (nên bật nếu pipeline dùng TF-IDF)
ESTIMATE_IDF = True
# ===========================


# sklearn chỉ dùng trong file build (không bắt buộc ở runtime nạp .npy)
from sklearn.cluster import MiniBatchKMeans
import joblib


def require_sift():
    """Tạo SIFT 128-D; báo lỗi rõ nếu OpenCV thiếu SIFT (không fallback ORB)."""
    try:
        sift = cv2.SIFT_create(nfeatures=1200)
    except Exception:
        sift = None
    if sift is None:
        raise RuntimeError(
            "OpenCV của bạn không có SIFT (opencv-contrib). "
            "Hãy cài opencv-contrib hoặc build OpenCV kèm SIFT."
        )
    return sift


def imread_gray_resized(path, resize_max):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if resize_max and resize_max > 0:
        h, w = img.shape[:2]
        s = float(resize_max) / max(h, w)
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)))
    return img


def sample_descriptors(paths, det, max_total, max_feats_per_img, resize_max):
    """Lấy mẫu SIFT descriptor tối đa max_total; giới hạn mỗi ảnh max_feats_per_img."""
    all_desc = []
    total = 0
    for p in tqdm(paths, desc="Extracting descriptors"):
        img = imread_gray_resized(p, resize_max)
        if img is None:
            continue
        kps, desc = det.detectAndCompute(img, None)
        if desc is None or len(desc) == 0:
            continue
        if len(desc) > max_feats_per_img:
            idx = np.random.choice(len(desc), max_feats_per_img, replace=False)
            desc = desc[idx]
        desc = desc.astype(np.float32, copy=False)
        all_desc.append(desc)
        total += len(desc)
        if total >= max_total:
            break
    if not all_desc:
        return np.empty((0, 128), np.float32)
    return np.vstack(all_desc)


def fit_codebook(desc, k, batch_size, seed, n_init):
    """Fit MiniBatchKMeans (không dùng n_init='auto' để tương thích sklearn cũ)."""
    km = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        reassignment_ratio=0.01,
        n_init=n_init,
        verbose=1,
        random_state=seed,
    )
    km.fit(desc)
    centers = km.cluster_centers_.astype(np.float32, copy=False)
    return km, centers


def assign_words(desc, centers):
    """Gán mỗi descriptor về trung tâm gần nhất bằng numpy (không phụ thuộc sklearn)."""
    if desc is None or len(desc) == 0:
        return None
    desc = desc.astype(np.float32, copy=False)  # (n,128)
    a2 = np.sum(desc * desc, axis=1, keepdims=True)                     # (n,1)
    b2 = np.sum(centers * centers, axis=1, keepdims=True).T             # (1,K)
    ab = desc @ centers.T                                               # (n,K)
    dist2 = a2 + b2 - 2.0 * ab                                          # (n,K)
    return np.argmin(dist2, axis=1)                                     # (n,)


def estimate_idf(paths, det, centers, resize_max):
    """
    IDF dựa theo df ảnh:
      df[j] = số ảnh có ít nhất 1 từ vựng j
      idf[j] = log((N+1)/(df[j]+1))
    """
    K = centers.shape[0]
    df = np.zeros(K, dtype=np.float32)
    for p in tqdm(paths, desc="Estimating IDF"):
        img = imread_gray_resized(p, resize_max)
        if img is None:
            continue
        kps, d = det.detectAndCompute(img, None)
        if d is None or len(d) == 0:
            continue
        words = assign_words(d, centers)
        if words is None:
            continue
        df[np.unique(words)] += 1
    N = float(len(paths))
    idf = np.log((N + 1.0) / (df + 1.0)).astype(np.float32)
    return idf


def main():
    np.random.seed(SEED)

    paths = sorted(glob.glob(IMG_GLOB, recursive=True))
    assert len(paths) > 0, f"Không tìm thấy ảnh với glob: {IMG_GLOB}"

    sift = require_sift()

    # 1) Lấy mẫu descriptor
    desc = sample_descriptors(paths, sift, MAX_TOTAL, MAX_FEATS_PER_IMG, RESIZE_MAX)
    print("Descriptors:", desc.shape)
    assert desc.shape[0] > 0 and desc.shape[1] == 128, "Descriptor rỗng hoặc không phải 128-D (SIFT)!"

    # 2) Fit codebook
    kmeans, centers = fit_codebook(desc, K, BATCH_SIZE, SEED, N_INIT)

    # 3) IDF (tuỳ chọn)
    if ESTIMATE_IDF:
        idf = estimate_idf(paths, sift, centers, RESIZE_MAX)
    else:
        idf = np.ones(K, dtype=np.float32)

    # 4) Lưu định dạng trung lập
    out_centers = f"{OUT_PREFIX}_centers.npy"
    out_idf = f"{OUT_PREFIX}_idf.npy"
    out_meta = f"{OUT_PREFIX}_meta.json"
    np.save(out_centers, centers)
    np.save(out_idf, idf)
    meta = dict(
        k=int(K),
        desc_dim=128,
        resize_max=int(RESIZE_MAX),
        max_total=int(MAX_TOTAL),
        batch_size=int(BATCH_SIZE),
        max_feats_per_img=int(MAX_FEATS_PER_IMG),
        seed=int(SEED),
    )
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {out_centers}, {out_idf}, {out_meta}")

    # 5) (tuỳ chọn) Lưu thêm bản sklearn nếu muốn dùng ngay cùng môi trường này
    if SAVE_JOBLIB:
        out_joblib = f"{OUT_PREFIX}_codebook.joblib"
        joblib.dump(kmeans, out_joblib)
        print(f"Also saved: {out_joblib}")


if __name__ == "__main__":
    main()
