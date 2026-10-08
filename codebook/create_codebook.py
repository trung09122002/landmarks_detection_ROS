# build_codebook.py
# thay doi K va batch_size neu thay doi max_total
import glob, cv2, numpy as np, joblib
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm

K = 4096               # số visual words (512–4096 tùy dữ liệu)
MAX_FEATS_PER_IMG = 800
IMG_GLOB = "../dataset/train/images/*.jpg"  # thư mục ảnh landmark
USE_SURF = False  # True nếu bạn có opencv-contrib và chấp nhận license SURF

def create_detector():
    if USE_SURF and hasattr(cv2, "xfeatures2d"):
        return cv2.xfeatures2d.SURF_create(hessianThreshold=400)
    return cv2.SIFT_create(nfeatures=1200)

def sample_desc(paths, max_total=2_000_000):
    det = create_detector()
    all_desc = []
    total = 0
    for p in tqdm(paths, desc="Extracting descriptors"):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        h, w = img.shape[:2]
        s = 640.0 / max(h, w)
        if s < 1.0: img = cv2.resize(img, (int(w*s), int(h*s)))
        kps, desc = det.detectAndCompute(img, None)
        if desc is None or len(desc)==0: continue
        if len(desc) > MAX_FEATS_PER_IMG:
            idx = np.random.choice(len(desc), MAX_FEATS_PER_IMG, replace=False)
            desc = desc[idx]
        all_desc.append(desc.astype(np.float32))
        total += len(desc)
        if total >= max_total: break
    return np.vstack(all_desc) if all_desc else np.empty((0,128), np.float32)

def estimate_idf(paths, kmeans):
    det = create_detector()
    K = kmeans.n_clusters
    df = np.zeros(K, dtype=np.float32)
    for p in tqdm(paths, desc="Estimating TF/IDF"):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        kps, d = det.detectAndCompute(img, None)
        if d is None or len(d)==0: continue
        words = kmeans.predict(d.astype(np.float32))
        df[np.unique(words)] += 1
    idf = np.log((len(paths)+1)/(df+1)).astype(np.float32)
    return idf

def main():
    paths = glob.glob(IMG_GLOB, recursive=True)
    assert len(paths)>0, "Không tìm thấy ảnh trong landmarks_train/"
    desc = sample_desc(paths)
    print("Descriptors:", desc.shape)
    assert desc.shape[0]>0, "Không thu được descriptor nào!"

    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=30_000, reassignment_ratio=0.01, verbose=1)
    kmeans.fit(desc)

    idf = estimate_idf(paths, kmeans)
    joblib.dump(kmeans, "codebook_kNN.joblib")
    np.save("idf.npy", idf)
    print("Saved: codebook_kNN.joblib, idf.npy")

if __name__ == "__main__":
    main()
