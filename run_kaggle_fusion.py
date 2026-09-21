import os
import sys
import time
import subprocess
import shutil
import re

print("=" * 65)
print("🚀 ĐANG KHỞI ĐỘNG MEDIA FUSION STUDIO TRÊN KAGGLE (2X T4 GPU)...")
print("=" * 65)

work_dir = "/kaggle/working"
app_dir = os.path.join(work_dir, "facefusion")
output_dir = os.path.join(work_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

# 1. Tải mã nguồn FaceFusion
print("\n⚡ [1/4] Chuẩn bị mã nguồn FaceFusion...")
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

# Vá share=True nếu có default.py
for root, dirs, files in os.walk(app_dir):
    for f in files:
        if f == 'default.py':
            fp = os.path.join(root, f)
            with open(fp, 'r', encoding='utf-8', errors='ignore') as rf:
                txt = rf.read()
            if 'launch(f' in txt and 'share=True' not in txt:
                txt = txt.replace('launch(f', 'launch(share=True, f')
                with open(fp, 'w', encoding='utf-8') as wf:
                    wf.write(txt)

# 3. Khởi động Cloudflare Tunnel
print("\n🌐 [3/4] Khởi tạo Cloudflare Tunnel công khai...", flush=True)
subprocess.run(['curl', '-LOs', 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb'], cwd=work_dir)
subprocess.run(['dpkg', '-i', os.path.join(work_dir, 'cloudflared-linux-amd64.deb')], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
log_path = os.path.join(work_dir, "cloudflared.log")
subprocess.Popen(['nohup', 'cloudflared', 'tunnel', '--url', 'http://127.0.0.1:7860'], stdout=open(log_path, 'w'), stderr=subprocess.STDOUT)

# Chờ link Cloudflare
cf_link = ""
for _ in range(10):
    time.sleep(1)
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            matches = re.findall(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com", f.read())
            if matches:
                cf_link = matches[0]
                break

# 4. Khởi chạy WebUI
print("\n" + "=" * 65, flush=True)
print("🎉 MEDIA FUSION STUDIO ĐÃ SẴN SÀNG TRÊN KAGGLE!", flush=True)
if cf_link:
    print(f"🔗 LINK TRUY CẬP WEBUI (CLOUDFLARE): {cf_link}", flush=True)
print("🔗 LINK GRADIO CŨNG SẼ XUẤT HIỆN NGAY BÊN DƯỚI:", flush=True)
print(f"📁 Video kết quả sẽ được lưu tại: {output_dir}", flush=True)
print("=" * 65 + "\n", flush=True)

subprocess.run([sys.executable, 'facefusion.py', 'run', '--execution-providers', 'cuda', '--output-path', output_dir], cwd=app_dir)
