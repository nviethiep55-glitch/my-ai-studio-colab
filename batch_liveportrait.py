import os
import sys
import time
import glob
import shutil
import subprocess

print("=" * 65)
print("🚀 LIVEPORTRAIT BATCH STUDIO - CHẾ ĐỘ CHẠY HÀNG LOẠT CÔNG NGHIỆP")
print("=" * 65)

# 1. Gắn kết Google Drive
print("\n🔗 [1/5] Kiểm tra kết nối Google Drive...")
has_drive = False
if os.path.exists('/content/drive/MyDrive'):
    print("  ✓ Google Drive đã gắn kết!")
    has_drive = True
else:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        has_drive = True
        print("  ✓ Đã gắn kết Google Drive thành công!")
    except Exception as e:
        print(f"  ⚠️ Không thể kết nối Drive tự động ({e}).")

# Đặt ngay trong thư mục LivePortrait quen thuộc của người dùng
base_dir = "/content/drive/MyDrive/AI_Colab_Cache/LivePortrait" if has_drive else "/content/LivePortrait_Batch"
input_dir = os.path.join(base_dir, "inputs")
driving_dir = os.path.join(base_dir, "driving")
output_dir = os.path.join(base_dir, "outputs")

os.makedirs(input_dir, exist_ok=True)
os.makedirs(driving_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

print(f"  📁 Thư mục chứa ảnh nhân vật : {input_dir}")
print(f"  📁 Thư mục chứa video cử động: {driving_dir}")
print(f"  📁 Thư mục lưu video kết quả : {output_dir}")

# 2. Kiểm tra mã nguồn và thư viện
print("\n⚡ [2/5] Chuẩn bị môi trường LivePortrait...")
os.chdir('/content')
if not os.path.exists('/content/LivePortrait'):
    subprocess.run(['git', 'clone', '-b', 'dev', 'https://github.com/camenduru/LivePortrait', '/content/LivePortrait'], check=True)
os.chdir('/content/LivePortrait')

def ensure_pkg(pkg_name, import_name=None):
    if import_name is None:
        import_name = pkg_name
    try:
        __import__(import_name)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg_name], check=True)

ensure_pkg("tyro")
ensure_pkg("gradio")
ensure_pkg("onnx")
ensure_pkg("onnxruntime-gpu", "onnxruntime")
ensure_pkg("colorama")
ensure_pkg("ffmpeg-python", "ffmpeg")
ensure_pkg("huggingface_hub")

# Vá lỗi PyTorch 2.6 toàn cục
import site
for p in site.getsitepackages():
    sc_file = os.path.join(p, 'sitecustomize.py')
    with open(sc_file, 'w', encoding='utf-8') as f:
        f.write('''import torch
_old_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _old_torch_load(*args, **kwargs)
torch.load = _safe_torch_load
''')
    break

subprocess.run(['git', 'checkout', 'src/utils/helper.py'], stderr=subprocess.DEVNULL)

# Nạp weights từ Drive Cache
drive_weights_dir = os.path.join(base_dir, "pretrained_weights")
if os.path.exists(drive_weights_dir) and os.path.exists(os.path.join(drive_weights_dir, "liveportrait/base_models/appearance_feature_extractor.pth")):
    if not os.path.exists('/content/LivePortrait/pretrained_weights'):
        print("  ✓ Nạp trọng số từ Google Drive Cache...")
        shutil.copytree(drive_weights_dir, '/content/LivePortrait/pretrained_weights')
else:
    if not os.path.exists('/content/LivePortrait/pretrained_weights'):
        from huggingface_hub import snapshot_download
        print("  ⏳ Tải trọng số từ HuggingFace...")
        snapshot_download(repo_id='camenduru/LivePortrait', local_dir='/content/LivePortrait/pretrained_weights', local_dir_use_symlinks=False)

# Biên dịch Cython
mesh_cython_dir = '/content/LivePortrait/src/utils/dependencies/insightface/thirdparty/face3d/mesh/cython'
if not glob.glob(os.path.join(mesh_cython_dir, '*.so')):
    os.chdir(mesh_cython_dir)
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], check=True)
    os.chdir('/content/LivePortrait')

# 3. Chuẩn bị dữ liệu mẫu nếu thư mục trống
image_exts = ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.PNG')
images = []
for ext in image_exts:
    images.extend(glob.glob(os.path.join(input_dir, ext)))

video_exts = ('*.mp4', '*.mov', '*.avi', '*.MP4', '*.MOV')
driving_videos = []
for ext in video_exts:
    driving_videos.extend(glob.glob(os.path.join(driving_dir, ext)))

# Nếu chưa có video lái, copy video mẫu từ assets
if not driving_videos:
    sample_video = '/content/LivePortrait/assets/examples/driving/d0.mp4'
    if os.path.exists(sample_video):
        dest_video = os.path.join(driving_dir, 'sample_driving.mp4')
        shutil.copy(sample_video, dest_video)
        driving_videos.append(dest_video)
        print(f"  ℹ️ Đã tự tạo 1 video lái mẫu tại: {dest_video}")

# Nếu chưa có ảnh, copy ảnh mẫu từ assets
if not images:
    sample_img = '/content/LivePortrait/assets/examples/source/s6.jpg'
    if os.path.exists(sample_img):
        dest_img = os.path.join(input_dir, 'sample_portrait.jpg')
        shutil.copy(sample_img, dest_img)
        images.append(dest_img)
        print(f"  ℹ️ Đã tự tạo 1 ảnh chân dung mẫu tại: {dest_img}")

selected_driving = driving_videos[0]
print(f"\n🎬 Video lái được chọn: {os.path.basename(selected_driving)}")
print(f"🖼️ Tổng số ảnh cần xử lý: {len(images)} ảnh")

# 4. Khởi tạo Pipeline LivePortrait DUY NHẤT 1 LẦN vào GPU VRAM
print("\n⚙️ [4/5] Khởi tạo mô hình AI vào GPU VRAM (Chỉ chạy 1 lần)...")
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

default_args = ArgumentConfig()
default_args.driving_info = selected_driving
default_args.flag_pasteback = False  # Tắt pasteback để render siêu tốc công nghiệp (~10-15s/video)
default_args.flag_do_crop = True

inference_cfg = partial_fields(InferenceConfig, default_args.__dict__)
crop_cfg = partial_fields(CropConfig, default_args.__dict__)

pipeline = LivePortraitPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)
print("  ✓ Mô hình AI đã nạp sẵn vào VRAM!")

# 5. Vòng lặp Render Hàng Loạt Siêu Tốc
print("\n🚀 [5/5] BẮT ĐẦU XỬ LÝ HÀNG LOẠT (BATCH PROCESSING)...")
total_start = time.time()

for idx, img_path in enumerate(images, 1):
    img_name = os.path.basename(img_path)
    stem = os.path.splitext(img_name)[0]
    out_name = f"result_{stem}.mp4"
    out_final_path = os.path.join(output_dir, out_name)
    
    print(f"\n[{idx}/{len(images)}] Đang xử lý: {img_name}...")
    start_t = time.time()
    
    args = ArgumentConfig()
    args.source_image = img_path
    args.driving_info = selected_driving
    args.output_dir = output_dir
    args.flag_pasteback = False
    args.flag_do_crop = True
    
    try:
        res = pipeline.execute(args)
        if isinstance(res, (list, tuple)) and len(res) > 0 and res[0] and os.path.exists(res[0]):
            shutil.copy(res[0], out_final_path)
        elapse = time.time() - start_t
        print(f"  ✅ Hoàn tất trong {elapse:.1f}s -> Đã lưu: {out_name}")
    except Exception as e:
        print(f"  ❌ Lỗi khi xử lý {img_name}: {e}")

total_elapse = time.time() - total_start
print("\n" + "=" * 65)
print(f"🎉 TẤT CẢ ĐÃ HOÀN TẤT! Tổng thời gian: {total_elapse:.1f}s")
print(f"📁 Toàn bộ video kết quả đã lưu tại Google Drive: {output_dir}")
print("=" * 65)
