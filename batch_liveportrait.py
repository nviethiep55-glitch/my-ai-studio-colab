import os
import sys
import time
import glob
import shutil
import subprocess

# Đảm bảo đường dẫn mã nguồn LivePortrait luôn nằm đầu sys.path
sys.path.insert(0, '/content/LivePortrait')

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
        print(f"  ⚠️ Chưa gắn kết Drive ({e}). Sẽ sử dụng bộ nhớ tạm /content.")

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

# 2. Chuẩn bị mã nguồn và thư viện
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
        print(f"  ⏳ Đang cài đặt thư viện {pkg_name}...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg_name], check=True)

print("  ⏳ Đang kiểm tra các gói thư viện phụ thuộc...", flush=True)
ensure_pkg("tyro")
ensure_pkg("gradio")
ensure_pkg("onnx")
ensure_pkg("onnxruntime-gpu", "onnxruntime")
ensure_pkg("colorama")
ensure_pkg("ffmpeg-python", "ffmpeg")
ensure_pkg("huggingface_hub")
ensure_pkg("cython")
ensure_pkg("setuptools")
print("  ✓ Các gói thư viện phụ thuộc đã sẵn sàng!", flush=True)

# Vá lỗi PyTorch 2.6 trong tiến trình hiện tại và toàn cục
try:
    import torch
    _old_torch_load = torch.load
    def _safe_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _old_torch_load(*args, **kwargs)
    torch.load = _safe_torch_load
except Exception:
    pass

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

# Vá lỗi import mesh_core_cython (không dùng trong LivePortrait 2D pipeline)
mesh_init_file = '/content/LivePortrait/src/utils/dependencies/insightface/thirdparty/face3d/mesh/__init__.py'
if os.path.exists(mesh_init_file):
    with open(mesh_init_file, 'r', encoding='utf-8') as f:
        mesh_init_code = f.read()
    if 'from .cython import mesh_core_cython' in mesh_init_code and 'try:' not in mesh_init_code:
        mesh_init_code = mesh_init_code.replace(
            'from .cython import mesh_core_cython',
            'try:\n    from .cython import mesh_core_cython\nexcept Exception:\n    mesh_core_cython = None'
        )
        with open(mesh_init_file, 'w', encoding='utf-8') as f:
            f.write(mesh_init_code)

# Nạp weights từ Drive Cache hoặc HuggingFace (Kiểm tra đầy đủ 8 file cốt lõi)
drive_weights_dir = os.path.join(base_dir, "pretrained_weights")
core_files = [
    "liveportrait/base_models/appearance_feature_extractor.pth",
    "liveportrait/base_models/motion_extractor.pth",
    "liveportrait/base_models/spade_generator.pth",
    "liveportrait/base_models/warping_module.pth",
    "liveportrait/retargeting_models/stitching_retargeting_module.pth",
    "liveportrait/landmark.onnx",
    "insightface/models/buffalo_l/det_10g.onnx",
    "insightface/models/buffalo_l/2d106det.onnx"
]

all_local_valid = all(
    os.path.exists(os.path.join('/content/LivePortrait/pretrained_weights', f)) and
    os.path.getsize(os.path.join('/content/LivePortrait/pretrained_weights', f)) > 100000
    for f in core_files
)

if not all_local_valid:
    all_drive_valid = all(
        os.path.exists(os.path.join(drive_weights_dir, f)) and
        os.path.getsize(os.path.join(drive_weights_dir, f)) > 100000
        for f in core_files
    )
    if all_drive_valid:
        print("  ✓ Nạp đầy đủ 8 file trọng số từ Google Drive Cache...", flush=True)
        shutil.copytree(drive_weights_dir, '/content/LivePortrait/pretrained_weights', dirs_exist_ok=True)
    else:
        from huggingface_hub import snapshot_download
        print("  ⏳ Đang tải toàn bộ 8 file trọng số AI thật (~650MB) từ HuggingFace...", flush=True)
        shutil.rmtree('/content/LivePortrait/pretrained_weights', ignore_errors=True)
        if os.path.exists(drive_weights_dir):
            shutil.rmtree(drive_weights_dir, ignore_errors=True)
        snapshot_download(repo_id='camenduru/LivePortrait', local_dir='/content/LivePortrait/pretrained_weights', local_dir_use_symlinks=False)
        if has_drive:
            print("  💾 Đang lưu bản sao trọng số hoàn chỉnh vào Google Drive để lần sau nạp ngay...", flush=True)
            os.makedirs(base_dir, exist_ok=True)
            shutil.copytree('/content/LivePortrait/pretrained_weights', drive_weights_dir, dirs_exist_ok=True)

# Biên dịch Cython nếu có thể
mesh_cython_dir = '/content/LivePortrait/src/utils/dependencies/insightface/thirdparty/face3d/mesh/cython'
if os.path.exists(mesh_cython_dir):
    os.chdir(mesh_cython_dir)
    for old_f in glob.glob('*.so') + glob.glob('mesh_core_cython.cpp') + glob.glob('mesh_core_cython.c'):
        try: os.remove(old_f)
        except Exception: pass
    shutil.rmtree('build', ignore_errors=True)
    try:
        subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], capture_output=True, timeout=60)
    except Exception:
        pass

# Luôn quay về thư mục gốc LivePortrait
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
    available_driving = glob.glob('/content/LivePortrait/assets/examples/driving/*.mp4')
    if available_driving:
        dest_video = os.path.join(driving_dir, 'sample_driving.mp4')
        shutil.copy(available_driving[0], dest_video)
        driving_videos.append(dest_video)
        print(f"  ℹ️ Đã tự tạo 1 video lái mẫu tại: {dest_video}")

# Nếu chưa có ảnh, copy ảnh mẫu từ assets
if not images:
    available_images = []
    for ext in image_exts:
        available_images.extend(glob.glob(f'/content/LivePortrait/assets/examples/source/{ext}'))
    if available_images:
        sample_img = available_images[0]
        dest_img = os.path.join(input_dir, f'sample_portrait{os.path.splitext(sample_img)[1]}')
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

try:
    pipeline = LivePortraitPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)
    print("  ✓ Mô hình AI đã nạp sẵn vào VRAM!", flush=True)
except Exception as e:
    import traceback
    print("  ❌ LỖI KHỞI TẠO PIPELINE LIVEPORTRAIT:", flush=True)
    traceback.print_exc()
    sys.exit(1)

# 5. Vòng lặp Render Hàng Loạt Siêu Tốc
print("\n🚀 [5/5] BẮT ĐẦU XỬ LÝ HÀNG LOẠT (BATCH PROCESSING)...", flush=True)
total_start = time.time()

for idx, img_path in enumerate(images, 1):
    img_name = os.path.basename(img_path)
    stem = os.path.splitext(img_name)[0]
    out_name = f"result_{stem}.mp4"
    out_final_path = os.path.join(output_dir, out_name)
    
    print(f"\n[{idx}/{len(images)}] Đang xử lý: {img_name}...", flush=True)
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
        print(f"  ✅ Hoàn tất trong {elapse:.1f}s -> Đã lưu: {out_name}", flush=True)
    except Exception as e:
        print(f"  ❌ Lỗi khi xử lý {img_name}: {e}", flush=True)

total_elapse = time.time() - total_start
print("\n" + "=" * 65, flush=True)
print(f"🎉 TẤT CẢ ĐÃ HOÀN TẤT! Tổng thời gian: {total_elapse:.1f}s", flush=True)
print(f"📁 Toàn bộ video kết quả đã lưu tại: {output_dir}", flush=True)
print("=" * 65, flush=True)
