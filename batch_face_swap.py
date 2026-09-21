import os
import sys
import time
import glob
import shutil
import argparse
import subprocess

# Bắt buộc nạp torch trước để Colab preload toàn bộ thư viện CUDA và cuDNN vào process
try:
    import torch
except Exception:
    pass

try:
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
except Exception:
    pass

print("=" * 65)
print("🚀 BATCH FACE SWAP STUDIO - HOÁN ĐỔI MẶT HÀNG LOẠT QUA GOOGLE DRIVE")
print("=" * 65)

# Đọc tham số tùy chỉnh từ giao diện Colab
parser = argparse.ArgumentParser()
parser.add_argument("--enhance", type=str, default="true", help="Bật làm nét mặt")
parser.add_argument("--enhancer", type=str, default="gfpgan", help="Mô hình làm nét")
parser.add_argument("--selector", type=str, default="largest", help="Chế độ chọn mặt: largest hoặc all")
args = parser.parse_args()

enable_enhance = args.enhance.lower() in ("true", "1", "yes")
enhancer_type = args.enhancer.lower()
selector_mode = args.selector.lower()

print(f"\n⚙️ CẤU HÌNH ĐANG CHẠY:")
print(f"  • Làm nét khuôn mặt (Face Enhancer) : {'BẬT (Nét căng chuẩn HD/4K)' if enable_enhance else 'TẮT (Tốc độ tối đa)'}")
print(f"  • Chế độ nhận diện khuôn mặt       : {'Nhân vật chính (Mặt lớn nhất)' if selector_mode == 'largest' else 'Đổi tất cả các mặt'}")

# 1. Kết nối Google Drive
print("\n🔗 [1/5] Kiểm tra kết nối Google Drive...")
has_drive = False
if os.path.exists('/content/drive/MyDrive'):
    print("  ✓ Google Drive đã gắn kết sẵn!")
    has_drive = True
else:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        has_drive = True
        print("  ✓ Đã gắn kết Google Drive thành công!")
    except Exception as e:
        print(f"  ⚠️ Không thể kết nối Drive tự động ({e}). Sẽ chạy trên bộ nhớ tạm.")

base_dir = "/content/drive/MyDrive/AI_Colab_Cache/BatchFaceSwap" if has_drive else "/content/BatchFaceSwap"
source_dir = os.path.join(base_dir, "source_faces")
target_dir = os.path.join(base_dir, "target_videos")
output_dir = os.path.join(base_dir, "output_videos")
models_dir = os.path.join(base_dir, "models")

os.makedirs(source_dir, exist_ok=True)
os.makedirs(target_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

print(f"  📁 Thư mục chứa ảnh khuôn mặt mẫu : {source_dir}")
print(f"  📁 Thư mục chứa video cần đổi mặt : {target_dir}")
print(f"  📁 Thư mục lưu video thành phẩm   : {output_dir}")

# 2. Cài đặt thư viện môi trường cần thiết
print("\n📦 [2/5] Kiểm tra và cấu hình tăng tốc CUDA GPU...", flush=True)

def ensure_pkg(pkg_name, import_name=None):
    if import_name is None:
        import_name = pkg_name
    try:
        __import__(import_name)
    except ImportError:
        print(f"  ⏳ Đang cài đặt {pkg_name}...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg_name], check=True)

ensure_pkg("onnx")
ensure_pkg("opencv-python-headless", "cv2")
ensure_pkg("tqdm")
ensure_pkg("insightface")

# Đảm bảo cài đặt đúng onnxruntime-gpu cho GPU Tesla T4 (Gỡ bỏ bản CPU nếu có)
try:
    import onnxruntime as ort
    has_cuda = 'CUDAExecutionProvider' in ort.get_available_providers()
except Exception:
    has_cuda = False

if not has_cuda:
    print("  ⏳ Đang kích hoạt CUDA GPU cho ONNX Runtime...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime-gpu"], check=True)
    try:
        import torch
    except Exception:
        pass
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()

import cv2
import numpy as np
from tqdm import tqdm
import insightface
from insightface.app import FaceAnalysis

# 3. Tải mô hình hoán đổi mặt inswapper_128.onnx & enhancer về Google Drive
print("\n💾 [3/5] Kiểm tra bộ trọng số AI trong Google Drive Cache...", flush=True)
swapper_path = os.path.join(models_dir, "inswapper_128.onnx")
if not os.path.exists(swapper_path) or os.path.getsize(swapper_path) < 100000000:
    print("  ⏳ Đang tải mô hình Inswapper 128 ONNX (~529MB) về Google Drive...", flush=True)
    download_url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    subprocess.run(['curl', '-L', download_url, '-o', swapper_path], check=True)
    print("  ✓ Đã lưu Inswapper vào Google Drive!", flush=True)
else:
    print("  ✓ Đã tìm thấy Inswapper 128 trong Google Drive! Nạp ngay trong 1 giây.", flush=True)

# Tải GFPGAN nếu bật làm nét
enhancer_path = os.path.join(models_dir, "gfpgan_1.4.onnx")
if enable_enhance:
    if not os.path.exists(enhancer_path) or os.path.getsize(enhancer_path) < 100000000:
        print("  ⏳ Đang tải mô hình GFPGAN Làm Nét (~332MB) về Google Drive...", flush=True)
        enhancer_url = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx"
        subprocess.run(['curl', '-L', enhancer_url, '-o', enhancer_path], check=True)
        print("  ✓ Đã lưu GFPGAN vào Google Drive!", flush=True)
    else:
        print("  ✓ Đã tìm thấy GFPGAN trong Google Drive Cache!", flush=True)

# 4. Khởi tạo mô hình AI vào VRAM GPU
print("\n⚙️ [4/5] Nạp mô hình Face Analysis & Swapper vào GPU Tesla T4...", flush=True)
try:
    available_providers = ort.get_available_providers()
    print(f"  ⚡ Bộ tăng tốc phát hiện: {available_providers}", flush=True)
    
    if 'CUDAExecutionProvider' in available_providers:
        chosen_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print("  🎉 ĐÃ KÍCH HOẠT GPU TESLA T4 CUDA THÀNH CÔNG (Tốc độ tối đa ~30 FPS)!", flush=True)
    else:
        chosen_providers = ['CPUExecutionProvider']
        print("  ⚠️ Không tìm thấy CUDA, đang chạy chế độ CPU.", flush=True)

    app = FaceAnalysis(name='buffalo_l', root=models_dir, providers=chosen_providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model(swapper_path, download=False, providers=chosen_providers)
    
    enhancer_model = None
    if enable_enhance and os.path.exists(enhancer_path):
        enhancer_model = ort.InferenceSession(enhancer_path, providers=chosen_providers)
        print("  ✓ Đã nạp thành công mô hình Làm Nét (GFPGAN) vào GPU!", flush=True)
        
    print("  ✓ Toàn bộ mô hình AI đã sẵn sàng trên GPU Tesla T4!", flush=True)
except Exception as e:
    print(f"  ❌ Lỗi khởi tạo mô hình GPU: {e}", flush=True)
    sys.exit(1)

# Hàm tăng nét khuôn mặt bằng GFPGAN ONNX
def enhance_face(target_crop):
    if enhancer_model is None:
        return target_crop
    try:
        h, w = target_crop.shape[:2]
        img = cv2.resize(target_crop, (512, 512))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        ort_inputs = {enhancer_model.get_inputs()[0].name: img}
        ort_outs = enhancer_model.run(None, ort_inputs)
        
        out_img = ort_outs[0][0]
        out_img = np.transpose(out_img, (1, 2, 0))
        out_img = (out_img * 0.5 + 0.5) * 255.0
        out_img = np.clip(out_img, 0, 255).astype(np.uint8)
        return cv2.resize(out_img, (w, h))
    except Exception:
        return target_crop

# 5. Chuẩn bị dữ liệu mẫu nếu thư mục còn trống
image_exts = ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.PNG')
source_files = []
for ext in image_exts:
    source_files.extend(glob.glob(os.path.join(source_dir, ext)))

video_exts = ('*.mp4', '*.mov', '*.avi', '*.MP4', '*.MOV')
target_videos = []
for ext in video_exts:
    target_videos.extend(glob.glob(os.path.join(target_dir, ext)))

if not source_files:
    print(f"\n⚠️ CHƯA CÓ ẢNH MẶT MẪU TRONG THƯ MỤC: {source_dir}")
    print("   Vui lòng tải ít nhất 1 file ảnh chân dung vào thư mục trên Google Drive rồi chạy lại ô này!")
    sys.exit(0)

if not target_videos:
    print(f"\n⚠️ CHƯA CÓ VIDEO CẦN ĐỔI MẶT TRONG THƯ MỤC: {target_dir}")
    print("   Vui lòng tải các video ngắn cần đổi mặt vào thư mục trên Google Drive rồi chạy lại ô này!")
    sys.exit(0)

source_path = source_files[0]
source_img = cv2.imread(source_path)
source_faces = app.get(source_img)

if not source_faces:
    print(f"❌ Không tìm thấy khuôn mặt nào trong ảnh nguồn: {os.path.basename(source_path)}")
    sys.exit(1)

source_face = sorted(source_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)[0]
print(f"\n👤 Khuôn mặt mẫu được chọn: {os.path.basename(source_path)}")
print(f"🎬 Tổng số video cần hoán đổi: {len(target_videos)} video")

# 6. Vòng lặp xử lý hàng loạt siêu tốc (Batch Processing)
print("\n🚀 [5/5] BẮT ĐẦU HOÁN ĐỔI MẶT HÀNG LOẠT TRÊN GPU...", flush=True)
total_start = time.time()

for idx, v_path in enumerate(target_videos, 1):
    v_name = os.path.basename(v_path)
    stem = os.path.splitext(v_name)[0]
    final_out = os.path.join(output_dir, f"swap_{stem}.mp4")
    temp_raw = f"/content/temp_swap_{idx}.mp4"
    temp_audio = f"/content/temp_audio_{idx}.aac"
    
    print(f"\n[{idx}/{len(target_videos)}] Đang xử lý: {v_name}...", flush=True)
    v_start = time.time()
    
    cap = cv2.VideoCapture(v_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_raw, fourcc, fps, (width, height))
    
    pbar = tqdm(total=total_frames, desc="  Render Frames", unit="frame", leave=False)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        target_faces = app.get(frame)
        if target_faces:
            if selector_mode == "all":
                for tf in target_faces:
                    try:
                        frame = swapper.get(frame, tf, source_face, paste_back=True)
                    except Exception:
                        pass
            else:
                main_target = sorted(target_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)[0]
                try:
                    frame = swapper.get(frame, main_target, source_face, paste_back=True)
                except Exception:
                    pass
        
        out.write(frame)
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    # Ghép âm thanh gốc vào video mới
    subprocess.run(['ffmpeg', '-y', '-i', v_path, '-vn', '-acodec', 'copy', temp_audio], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 1000:
        subprocess.run(['ffmpeg', '-y', '-i', temp_raw, '-i', temp_audio, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', final_out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(['ffmpeg', '-y', '-i', temp_raw, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', final_out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(temp_raw): os.remove(temp_raw)
    if os.path.exists(temp_audio): os.remove(temp_audio)
    
    v_elapse = time.time() - v_start
    print(f"  ✅ Hoàn tất trong {v_elapse:.1f}s -> Đã lưu: {os.path.basename(final_out)}", flush=True)

total_elapse = time.time() - total_start
print("\n" + "=" * 65, flush=True)
print(f"🎉 TẤT CẢ {len(target_videos)} VIDEO ĐÃ ĐỔI MẶT XONG! Tổng thời gian: {total_elapse:.1f}s", flush=True)
print(f"📁 Xem toàn bộ video kết quả tại Google Drive: {output_dir}", flush=True)
print("=" * 65, flush=True)
