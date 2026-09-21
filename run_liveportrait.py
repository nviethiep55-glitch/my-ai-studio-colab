import os
import sys
import time
import shutil
import subprocess

print("=" * 65)
print("🚀 ĐANG KHỞI ĐỘNG LIVEPORTRAIT STUDIO VỚI GOOGLE DRIVE CACHING...")
print("=" * 65)

# 1. Gắn kết Google Drive thông minh
print("\n🔗 [1/6] Kiểm tra kết nối Google Drive...")
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
        print(f"  ⚠️ Không thể kết nối Drive tự động ({e}). Sẽ chạy trên bộ nhớ tạm của Colab.")

drive_cache_dir = "/content/drive/MyDrive/AI_Colab_Cache/LivePortrait" if has_drive else "/content/cache/LivePortrait"
drive_weights_dir = f"{drive_cache_dir}/pretrained_weights"
os.makedirs(drive_cache_dir, exist_ok=True)

# 2. Tải mã nguồn LivePortrait
print("\n⚡ [2/6] Chuẩn bị mã nguồn LivePortrait...")
os.chdir('/content')
if not os.path.exists('/content/LivePortrait'):
    subprocess.run(['git', 'clone', '-b', 'dev', 'https://github.com/camenduru/LivePortrait', '/content/LivePortrait'], check=True)
os.chdir('/content/LivePortrait')

# 3. Cài đặt các thư viện môi trường cần thiết
print("\n📦 [3/6] Cài đặt và kiểm tra thư viện môi trường...")

def ensure_pkg(pkg_name, import_name=None):
    if import_name is None:
        import_name = pkg_name
    try:
        __import__(import_name)
        print(f"  ✓ {import_name}: Sẵn sàng")
    except ImportError:
        print(f"  ⏳ Đang cài đặt {pkg_name}...")
        res = subprocess.run([sys.executable, "-m", "pip", "install", pkg_name], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"  ❌ Lỗi khi cài đặt {pkg_name}:\n{res.stderr}")
            raise RuntimeError(f"Cài đặt {pkg_name} thất bại!")
        __import__(import_name)
        print(f"  ✓ {import_name}: Cài đặt thành công")

ensure_pkg("tyro")
ensure_pkg("gradio")
ensure_pkg("onnx")
ensure_pkg("onnxruntime-gpu", "onnxruntime")
ensure_pkg("colorama")
ensure_pkg("ffmpeg-python", "ffmpeg")
ensure_pkg("huggingface_hub")

# 4. Vá lỗi PyTorch 2.6 toàn cục qua sitecustomize.py
print("\n🔧 [4/6] Cấu hình vá lỗi PyTorch 2.6 toàn cục...")
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

# Phục hồi nguyên bản helper.py nếu trước đó bị can thiệp
subprocess.run(['git', 'checkout', 'src/utils/helper.py'], stderr=subprocess.DEVNULL)

# 5. Kiểm tra toàn bộ 8 file trọng số cốt lõi
print("\n💾 [5/6] Kiểm tra bộ trọng số AI...")
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

all_drive_weights_valid = True
for f in core_files:
    target = os.path.join(drive_weights_dir, f)
    if not os.path.exists(target) or os.path.getsize(target) < 100000:
        all_drive_weights_valid = False
        break

if all_drive_weights_valid:
    print("  🎉 ĐÃ TÌM THẤY ĐỦ TOÀN BỘ 8 FILE TRỌNG SỐ THẬT TRONG GOOGLE DRIVE! Nạp trực tiếp trong 3 giây...")
    shutil.rmtree('/content/LivePortrait/pretrained_weights', ignore_errors=True)
    shutil.copytree(drive_weights_dir, '/content/LivePortrait/pretrained_weights')
else:
    print("  ⏳ Đang tải toàn bộ trọng số AI thật (~650MB) từ HuggingFace...")
    shutil.rmtree('/content/LivePortrait/pretrained_weights', ignore_errors=True)
    if os.path.exists(drive_weights_dir):
        shutil.rmtree(drive_weights_dir, ignore_errors=True)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id='camenduru/LivePortrait', local_dir='/content/LivePortrait/pretrained_weights', local_dir_use_symlinks=False)
    if has_drive:
        print("  💾 Đang lưu bản sao trọng số thật vào Google Drive để lần sau nạp ngay lập tức...")
        os.makedirs(drive_cache_dir, exist_ok=True)
        shutil.copytree('/content/LivePortrait/pretrained_weights', drive_weights_dir, dirs_exist_ok=True)

# 6. Biên dịch module Cython 3D
print("\n⚙️ [6/6] Biên dịch Cython và khởi động WebUI...")
os.chdir('/content/LivePortrait/src/utils/dependencies/insightface/thirdparty/face3d/mesh/cython')
subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], check=True)
os.chdir('/content/LivePortrait')

# Khởi động Cloudflare Tunnel dự phòng
subprocess.run(['curl', '-LOs', 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb'])
subprocess.run(['dpkg', '-i', 'cloudflared-linux-amd64.deb'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.Popen(['nohup', 'cloudflared', 'tunnel', '--url', 'http://127.0.0.1:8890'], stdout=open('/content/cloudflared.log', 'w'), stderr=subprocess.STDOUT)

time.sleep(4)
cf_link = ""
if os.path.exists("/content/cloudflared.log"):
    import re
    with open("/content/cloudflared.log", "r", encoding="utf-8", errors="ignore") as f:
        matches = re.findall(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com", f.read())
        if matches:
            cf_link = matches[0]

with open('/content/LivePortrait/app.py', 'r', encoding='utf-8') as f:
    app_text = f.read()

if 'while True: time.sleep(3600)' not in app_text:
    app_text += '\nimport time\nwhile True:\n    time.sleep(3600)\n'
    with open('/content/LivePortrait/app.py', 'w', encoding='utf-8') as f:
        f.write(app_text)

print("\n" + "=" * 65)
print("🎉 LIVEPORTRAIT WEBUI ĐANG HOẠT ĐỘNG!")
if cf_link:
    print(f"🔗 LINK CLOUDFLARE (TRUY CẬP NGAY): {cf_link}")
print("🔗 LINK GRADIO LIVE SẼ XUẤT HIỆN NGAY BÊN DƯỚI:")
print("=" * 65 + "\n")

subprocess.run([sys.executable, '-u', 'app.py', '--share', '--server_port', '8890'])
