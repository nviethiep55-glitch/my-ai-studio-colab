import os
import sys
import time
import subprocess
import shutil
import re
import threading
import socket

print("=" * 65)
print("🚀 ĐANG KHỞI ĐỘNG MEDIA FUSION STUDIO TRÊN KAGGLE (2X T4 GPU)...")
print("=" * 65)

work_dir = "/kaggle/working"
app_dir = os.path.join(work_dir, "facefusion")
output_dir = os.path.join(work_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

# 1. Tải mã nguồn FaceFusion
print("\n⚡ [1/4] Chuẩn bị mã nguồn FaceFusion...", flush=True)
if not os.path.exists(app_dir):
    subprocess.run(['git', 'clone', 'https://github.com/facefusion/facefusion.git', app_dir, '--single-branch'], check=True)

# 2. Cài đặt thư viện với CUDA 12
print("\n📦 [2/4] Cài đặt thư viện và cấu hình CUDA GPU...", flush=True)
try:
    subprocess.run([sys.executable, 'install.py', 'cuda@12', '--skip-conda'], cwd=app_dir, check=True)
except Exception:
    try:
        subprocess.run([sys.executable, 'install.py', '--onnxruntime', 'cuda@12', '--skip-conda'], cwd=app_dir, check=True)
    except Exception:
        subprocess.run([sys.executable, 'install.py', 'default', '--skip-conda'], cwd=app_dir, check=True)

# 3. Vá share=True trực tiếp vào tất cả file layouts để Gradio tạo link công khai
print("\n🔧 [3/4] Cấu hình Gradio Live Public Share...", flush=True)
patched = False
for root, dirs, files in os.walk(app_dir):
    for f in files:
        if f.endswith('.py'):
            fp = os.path.join(root, f)
            with open(fp, 'r', encoding='utf-8', errors='ignore') as rf:
                txt = rf.read()
            if '.launch(' in txt and 'share=True' not in txt and 'share = True' not in txt:
                txt = txt.replace('.launch(', '.launch(share=True, ')
                with open(fp, 'w', encoding='utf-8') as wf:
                    wf.write(txt)
                patched = True

if patched:
    print("  ✓ Đã kích hoạt Gradio Live Share tự động!", flush=True)

# 4. Tiến trình chạy ngầm Cloudflare Tunnel (Chỉ kết nối KHI cổng 7860 đã thực sự mở)
def start_tunnel_when_ready():
    for _ in range(300):
        try:
            with socket.create_connection(('127.0.0.1', 7860), timeout=1):
                break
        except OSError:
            time.sleep(1)
    
    # Khi cổng 7860 đã sẵn sàng, khởi tạo Cloudflare Tunnel
    subprocess.run(['curl', '-LOs', 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb'], cwd=work_dir)
    subprocess.run(['dpkg', '-i', os.path.join(work_dir, 'cloudflared-linux-amd64.deb')], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log_path = os.path.join(work_dir, "cloudflared.log")
    subprocess.Popen(['cloudflared', 'tunnel', '--url', 'http://127.0.0.1:7860'], stdout=open(log_path, 'w'), stderr=subprocess.STDOUT)
    
    for _ in range(15):
        time.sleep(1)
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                matches = re.findall(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com", f.read())
                if matches:
                    print("\n" + "=" * 65, flush=True)
                    print(f"🔗 LINK CLOUDFLARE (DỰ PHÒNG): {matches[0]}", flush=True)
                    print("=" * 65 + "\n", flush=True)
                    break

tunnel_thread = threading.Thread(target=start_tunnel_when_ready, daemon=True)
tunnel_thread.start()

# 5. Khởi chạy WebUI chính
print("\n🌐 [4/4] Khởi động FaceFusion WebUI...", flush=True)
print(f"📁 Video kết quả sẽ được lưu tại: {output_dir}\n", flush=True)
subprocess.run([sys.executable, 'facefusion.py', 'run', '--execution-providers', 'cuda', '--output-path', output_dir], cwd=app_dir)
