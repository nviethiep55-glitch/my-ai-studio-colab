import os
import sys
import time
import glob
import shutil
import queue
import argparse
import threading
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
print("🔥 HỖ TRỢ ĐỔI CHÍNH XÁC TỪNG NGƯỜI TRONG CLIP NHIỀU NHÂN VẬT")
print("=" * 65)

# Đọc tham số tùy chỉnh từ giao diện Colab
parser = argparse.ArgumentParser()
parser.add_argument("--enhance", type=str, default="false", help="Bật làm nét mặt")
parser.add_argument("--enhancer", type=str, default="gfpgan", help="Mô hình làm nét")
parser.add_argument("--selector", type=str, default="largest", help="Chế độ chọn mặt khi không có ảnh mẫu: largest hoặc all")
parser.add_argument("--speed", type=str, default="turbo", help="Chế độ tốc độ: turbo hoặc standard")
parser.add_argument("--mask_blur", type=float, default=0.18, help="Độ làm mềm viền mặt (0.05 - 0.30)")
parser.add_argument("--blend", type=float, default=0.80, help="Độ hòa trộn làm nét mặt (0.4 - 1.0)")
parser.add_argument("--crf", type=int, default=18, help="Chất lượng nén video CRF (16 - 24)")
args = parser.parse_args()

enable_enhance = args.enhance.lower() in ("true", "1", "yes")
enhancer_type = args.enhancer.lower()
selector_mode = args.selector.lower()
is_turbo = args.speed.lower() == "turbo"
mask_blur_val = args.mask_blur
enhancer_blend = args.blend
video_crf = args.crf

print(f"\n⚙️ CẤU HÌNH ĐANG CHẠY:")
print(f"  • Chế độ tăng tốc GPU              : {'🔥 TURBO KỊCH KHUNG (Đa luồng ~35-40 FPS)' if is_turbo else 'TIÊU CHUẨN (~20 FPS)'}")
print(f"  • Làm nét khuôn mặt (Face Enhancer): {'BẬT (Nét căng chuẩn HD/4K)' if enable_enhance else 'TẮT (Tốc độ tối đa ~7 phút/15k frame)'}")
print(f"  • Hòa trộn viền mềm (Mask Blur)    : {int(mask_blur_val * 100)}% (Tiệp da cổ & trán)")

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

# Thư mục chứa ảnh mặt mới & ảnh người trong clip (hỗ trợ cả tiếng Việt lẫn tiếng Anh)
mat_moi_dir = os.path.join(base_dir, "mat_moi")
source_dir = os.path.join(base_dir, "source_faces")
nguoi_trong_clip_dir = os.path.join(base_dir, "nguoi_trong_clip")
target_faces_dir = os.path.join(base_dir, "target_faces")
target_dir = os.path.join(base_dir, "target_videos")
output_dir = os.path.join(base_dir, "output_videos")
models_dir = os.path.join(base_dir, "models")

for d in [mat_moi_dir, source_dir, nguoi_trong_clip_dir, target_faces_dir, target_dir, output_dir, models_dir]:
    os.makedirs(d, exist_ok=True)

print(f"  📁 Thư mục [Mặt Mới muốn thay]        : {mat_moi_dir}")
print(f"  📁 Thư mục [Người trong clip cần đổi] : {nguoi_trong_clip_dir} (Tùy chọn khi clip có nhiều người)")
print(f"  📁 Thư mục chứa video cần đổi mặt     : {target_dir}")
print(f"  📁 Thư mục lưu video thành phẩm       : {output_dir}")

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
    print("  ⏳ Đang kích hoạt CUDA GPU cho ONNX Runtime trên Tesla T4...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime-gpu", "--extra-index-url", "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nvidia-cublas-cu12", "nvidia-cudnn-cu12"], check=False)
    print("  🚀 Tự động nạp nhân CUDA GPU vào tiến trình mới...", flush=True)
    os.execv(sys.executable, [sys.executable] + sys.argv)

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
        enhancer_url = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx"
        subprocess.run(['curl', '-L', enhancer_url, '-o', enhancer_path], check=True)
        print("  ✓ Đã lưu GFPGAN vào Google Drive!", flush=True)
    else:
        print("  ✓ Đã tìm thấy GFPGAN trong Google Drive Cache!", flush=True)

# 4. Khởi tạo mô hình AI vào VRAM GPU với CUDA Provider Tối ưu
print("\n⚙️ [4/5] Nạp mô hình Face Analysis & Swapper vào GPU Tesla T4...", flush=True)
try:
    available_providers = ort.get_available_providers()
    print(f"  ⚡ Bộ tăng tốc phát hiện: {available_providers}", flush=True)
    
    if 'CUDAExecutionProvider' in available_providers:
        cuda_options = {
            'device_id': 0,
            'arena_extend_strategy': 'kSameAsRequested',
            'gpu_mem_limit': 12 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'DEFAULT',
            'do_copy_in_default_stream': True,
        }
        chosen_providers = [('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider']
        print("  🎉 ĐÃ KÍCH HOẠT GPU TESLA T4 CUDA THÀNH CÔNG (Tốc độ tối đa ~35-40 FPS)!", flush=True)
    else:
        chosen_providers = ['CPUExecutionProvider']
        print("  ⚠️ Không tìm thấy CUDA, đang chạy chế độ CPU.", flush=True)

    det_size = (512, 512) if is_turbo else (640, 640)
    app = FaceAnalysis(name='buffalo_l', root=models_dir, providers=chosen_providers)
    app.prepare(ctx_id=0, det_size=det_size)
    swapper = insightface.model_zoo.get_model(swapper_path, download=False, providers=chosen_providers)
    
    enhancer_model = None
    if enable_enhance and os.path.exists(enhancer_path):
        enhancer_model = ort.InferenceSession(enhancer_path, providers=chosen_providers)
        print("  ✓ Đã nạp thành công mô hình Làm Nét (GFPGAN) vào GPU!", flush=True)
        
    print("  ✓ Toàn bộ mô hình AI đã sẵn sàng trên GPU Tesla T4!", flush=True)
except Exception as e:
    print(f"  ❌ Lỗi khởi tạo mô hình GPU: {e}", flush=True)
    sys.exit(1)

# Hàm tính độ tương đồng Cosine giữa 2 vector khuôn mặt
def compute_sim(emb1, emb2):
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-6))

def paste_back_seamless(target_img, bgr_fake, M, mask_blur_pct=0.18):
    """
    Seamless feather paste-back with elliptical soft mask.
    Triệt tiêu hoàn toàn viền cắt dán cứng, hòa trộn tự nhiên vào da cổ/trán.
    """
    h, w = bgr_fake.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.44), int(h * 0.47)), 0, 0, 360, 1.0, -1)
    
    ksize = max(3, int(w * mask_blur_pct) | 1)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    
    IM = cv2.invertAffineTransform(M)
    warped_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderMode=cv2.BORDER_CONSTANT)
    warped_mask = cv2.warpAffine(mask, IM, (target_img.shape[1], target_img.shape[0]), borderMode=cv2.BORDER_CONSTANT)
    warped_mask = np.clip(warped_mask, 0.0, 1.0)[:, :, np.newaxis]
    
    return (warped_fake * warped_mask + target_img * (1.0 - warped_mask)).astype(np.uint8)

def enhance_face(enhancer_sess, face_bgr, blend=0.8):
    if enhancer_sess is None:
        return face_bgr
    try:
        orig_h, orig_w = face_bgr.shape[:2]
        inp = cv2.resize(face_bgr, (512, 512))
        inp = inp.astype(np.float32) / 255.0
        inp = inp[:, :, ::-1]  # BGR to RGB
        inp = (inp - 0.5) / 0.5  # [-1, 1]
        inp = np.transpose(inp, (2, 0, 1))
        inp = np.expand_dims(inp, axis=0)

        in_name = enhancer_sess.get_inputs()[0].name
        out_name = enhancer_sess.get_outputs()[0].name
        out = enhancer_sess.run([out_name], {in_name: inp})[0][0]

        out = np.clip((out + 1.0) / 2.0, 0.0, 1.0)
        out = np.transpose(out, (1, 2, 0))
        out = (out[:, :, ::-1] * 255.0).astype(np.uint8)
        out = cv2.resize(out, (orig_w, orig_h))

        if blend < 1.0:
            out = cv2.addWeighted(out, blend, face_bgr, 1.0 - blend, 0)
        return out
    except Exception:
        return face_bgr

def perform_face_swap(frame, target_face, source_face):
    """
    Hoán đổi khuôn mặt với viền mềm tiệp da và làm nét GFPGAN nếu được bật.
    Tự động fallback về paste_back nguyên bản nếu có lỗi.
    """
    try:
        bgr_fake, M = swapper.get(frame, target_face, source_face, paste_back=False)
        if enhancer_model is not None:
            bgr_fake = enhance_face(enhancer_model, bgr_fake, blend=enhancer_blend)
        return paste_back_seamless(frame, bgr_fake, M, mask_blur_pct=mask_blur_val)
    except Exception:
        return swapper.get(frame, target_face, source_face, paste_back=True)

# 5. Quét ảnh khuôn mặt & Thiết lập cặp hoán đổi thông minh
image_exts = ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.PNG')

def get_images(folder):
    files = []
    for ext in image_exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(list(set(files)))

new_face_files = get_images(mat_moi_dir) + get_images(source_dir)
new_face_files = sorted(list(set(new_face_files)))

ref_face_files = get_images(nguoi_trong_clip_dir) + get_images(target_faces_dir)
ref_face_files = sorted(list(set(ref_face_files)))

video_exts = ('*.mp4', '*.mov', '*.avi', '*.MP4', '*.MOV', '*.webm')
target_videos = []
for ext in video_exts:
    target_videos.extend(glob.glob(os.path.join(target_dir, ext)))

if not new_face_files:
    print(f"\n⚠️ CHƯA CÓ ẢNH MẶT MỚI TRONG THƯ MỤC: {mat_moi_dir}")
    print("   Vui lòng tải ít nhất 1 file ảnh chân dung mặt mới muốn thay vào thư mục trên Google Drive rồi chạy lại ô này!")
    sys.exit(0)

if not target_videos:
    print(f"\n⚠️ CHƯA CÓ VIDEO CẦN ĐỔI MẶT TRONG THƯ MỤC: {target_dir}")
    print("   Vui lòng tải các video cần đổi mặt vào thư mục trên Google Drive rồi chạy lại ô này!")
    sys.exit(0)

swap_pairs = []
default_source_face = None

if ref_face_files:
    print(f"\n🎯 [CHẾ ĐỘ CHỈ ĐỊNH ĐÍCH DANH NHÂN VẬT - TARGET REFERENCE MATCH]")
    print(f"   Phát hiện {len(ref_face_files)} ảnh mẫu người trong clip cần đổi.")
    
    # Tạo map stem -> path của mặt mới
    new_faces_map = {os.path.splitext(os.path.basename(f))[0].lower(): f for f in new_face_files}
    
    for r_path in ref_face_files:
        r_stem = os.path.splitext(os.path.basename(r_path))[0].lower()
        r_img = cv2.imread(r_path)
        if r_img is None:
            continue
        r_faces = app.get(r_img)
        if not r_faces:
            print(f"  ⚠️ Không phát hiện khuôn mặt trong ảnh người trong clip: {os.path.basename(r_path)} -> Bỏ qua")
            continue
        ref_face = sorted(r_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)[0]
        
        # Ghép cặp với ảnh mặt mới tương ứng
        matched_new_path = None
        if r_stem in new_faces_map:
            matched_new_path = new_faces_map[r_stem]
        elif len(new_face_files) == 1:
            matched_new_path = new_face_files[0]
        else:
            idx = len(swap_pairs)
            matched_new_path = new_face_files[idx] if idx < len(new_face_files) else new_face_files[0]
            
        n_img = cv2.imread(matched_new_path)
        if n_img is None:
            continue
        n_faces = app.get(n_img)
        if not n_faces:
            print(f"  ⚠️ Không phát hiện khuôn mặt trong ảnh mặt mới: {os.path.basename(matched_new_path)} -> Bỏ qua")
            continue
        new_face = sorted(n_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)[0]
        
        swap_pairs.append({
            'ref_embedding': ref_face.embedding,
            'source_face': new_face,
            'ref_name': os.path.basename(r_path),
            'new_name': os.path.basename(matched_new_path)
        })
        print(f"  🔗 CẶP #{len(swap_pairs)}: [Người trong clip: {os.path.basename(r_path)}] ➡️ [Mặt mới: {os.path.basename(matched_new_path)}]")
else:
    print(f"\n👥 [CHẾ ĐỘ TỰ ĐỘNG THEO NHÂN VẬT CHÍNH]")
    print(f"   (Thư mục 'nguoi_trong_clip' đang trống -> Tự động đổi nhân vật chính hoặc tất cả)")
    default_new_path = new_face_files[0]
    n_img = cv2.imread(default_new_path)
    n_faces = app.get(n_img)
    if not n_faces:
        print(f"❌ Không tìm thấy khuôn mặt nào trong ảnh mặt mới: {os.path.basename(default_new_path)}")
        sys.exit(1)
    default_source_face = sorted(n_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)[0]
    print(f"  👤 Khuôn mặt mẫu được chọn: {os.path.basename(default_new_path)}")

print(f"\n🎬 Tổng số video cần hoán đổi: {len(target_videos)} video")

# 6. Xử lý video bằng ĐƯỜNG ỐNG ĐA LUỒNG (Turbo Pipeline)
def process_video_pipeline(v_path, final_out):
    cap = cv2.VideoCapture(v_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return False
        
    temp_raw = f"/content/temp_swap_{int(time.time()*1000)}.mp4"
    temp_audio = f"/content/temp_audio_{int(time.time()*1000)}.aac"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_raw, fourcc, fps, (width, height))
    
    # Hàng đợi bộ đệm đa luồng (Prefetch Queue)
    in_queue = queue.Queue(maxsize=32)
    out_queue = queue.Queue(maxsize=32)
    
    # Luồng 1: Đọc frame trước từ ổ đĩa vào RAM (Prefetch Thread)
    def reader_worker():
        f_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            in_queue.put((f_idx, frame))
            f_idx += 1
        in_queue.put(None)
        cap.release()
        
    # Luồng 2: Ghi frame đã swap ra đĩa ở luồng riêng (Writer Thread)
    def writer_worker():
        while True:
            item = out_queue.get()
            if item is None:
                break
            _, out_frame = item
            writer.write(out_frame)
            out_queue.task_done()
        writer.release()
        
    t_reader = threading.Thread(target=reader_worker, daemon=True)
    t_writer = threading.Thread(target=writer_worker, daemon=True)
    t_reader.start()
    t_writer.start()
    
    # Luồng 3: GPU Inference Engine (Chạy liên tục không bị nghẽn I/O)
    desc_label = f"  Render [{os.path.basename(v_path)[:20]}]"
    pbar = tqdm(total=total_frames, desc=desc_label, unit="frame", leave=False)
    
    while True:
        item = in_queue.get()
        if item is None:
            break
        f_idx, frame = item
        
        try:
            target_faces = app.get(frame)
            if target_faces:
                if swap_pairs:
                    # Chế độ CHỈ ĐỊNH ĐÍCH DANH theo ảnh mẫu (Target Reference Matching)
                    used_indices = set()
                    for pair in swap_pairs:
                        best_sim = -1.0
                        best_idx = -1
                        for t_idx, tf in enumerate(target_faces):
                            if t_idx in used_indices:
                                continue
                            sim = compute_sim(pair['ref_embedding'], tf.embedding)
                            if sim > best_sim:
                                best_sim = sim
                                best_idx = t_idx
                        # Ngưỡng chuẩn xác cùng 1 người (0.40)
                        if best_idx >= 0 and best_sim >= 0.40:
                            used_indices.add(best_idx)
                            frame = perform_face_swap(frame, target_faces[best_idx], pair['source_face'])
                else:
                    # Chế độ TỰ ĐỘNG khi không có ảnh mẫu người trong clip
                    if selector_mode == "all":
                        for tf in target_faces:
                            frame = perform_face_swap(frame, tf, default_source_face)
                    else:
                        main_target = sorted(target_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)[0]
                        frame = perform_face_swap(frame, main_target, default_source_face)
        except Exception:
            pass
            
        out_queue.put((f_idx, frame))
        pbar.update(1)
        
    pbar.close()
    out_queue.put(None)
    t_reader.join()
    t_writer.join()
    
    # Ghép âm thanh nguyên gốc vào video mới
    subprocess.run(['ffmpeg', '-y', '-i', v_path, '-vn', '-acodec', 'copy', temp_audio], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 1000:
        subprocess.run(['ffmpeg', '-y', '-i', temp_raw, '-i', temp_audio, '-c:v', 'libx264', '-crf', str(video_crf), '-preset', 'fast', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', final_out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(['ffmpeg', '-y', '-i', temp_raw, '-c:v', 'libx264', '-crf', str(video_crf), '-preset', 'fast', '-pix_fmt', 'yuv420p', final_out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    if os.path.exists(temp_raw):
        try: os.remove(temp_raw)
        except Exception: pass
    if os.path.exists(temp_audio):
        try: os.remove(temp_audio)
        except Exception: pass
        
    return True

# 7. Vòng lặp xử lý hàng loạt siêu tốc (Batch Loop)
print("\n🚀 [5/5] BẮT ĐẦU HOÁN ĐỔI MẶT HÀNG LOẠT TRÊN GPU TURBO PIPELINE...", flush=True)
total_start = time.time()

for idx, v_path in enumerate(target_videos, 1):
    v_name = os.path.basename(v_path)
    stem = os.path.splitext(v_name)[0]
    final_out = os.path.join(output_dir, f"swap_{stem}.mp4")
    
    print(f"\n[{idx}/{len(target_videos)}] Đang xử lý: {v_name}...", flush=True)
    v_start = time.time()
    
    ok = process_video_pipeline(v_path, final_out)
    
    v_elapse = time.time() - v_start
    if ok and os.path.exists(final_out):
        cap_check = cv2.VideoCapture(final_out)
        f_cnt = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_check.release()
        avg_fps = f_cnt / v_elapse if v_elapse > 0 else 0
        print(f"  ✅ Hoàn tất {f_cnt} frame trong {v_elapse:.1f}s (~{avg_fps:.1f} FPS) -> Đã lưu: {os.path.basename(final_out)}", flush=True)
    else:
        print(f"  ❌ Lỗi xử lý video: {v_name}", flush=True)

total_elapse = time.time() - total_start
print("\n" + "=" * 65, flush=True)
print(f"🎉 TẤT CẢ {len(target_videos)} VIDEO ĐÃ ĐỔI MẶT XONG! Tổng thời gian: {total_elapse:.1f}s", flush=True)
print(f"📁 Xem toàn bộ video kết quả tại Google Drive: {output_dir}", flush=True)
print("=" * 65, flush=True)
