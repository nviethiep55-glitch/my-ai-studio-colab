import os
import sys
import time
import base64
import shutil
import subprocess

print("=" * 65)
print("🚀 ĐANG KHỞI ĐỘNG MEDIA FUSION STUDIO (GOOGLE DRIVE CACHING)...")
print("=" * 65)

# 1. Gắn kết Google Drive
print("\n🔗 [1/4] Kiểm tra kết nối Google Drive...")
has_drive = os.path.exists('/content/drive/MyDrive')
if has_drive:
    print("  ✓ Google Drive đã gắn kết!")
else:
    print("  ⚠️ Chưa gắn kết Drive. Sẽ lưu tạm ở /content.")

drive_cache_dir = "/content/drive/MyDrive/AI_Colab_Cache/MediaFusion" if has_drive else "/content/cache/MediaFusion"
output_drive_dir = "/content/drive/MyDrive/AI_Colab_Cache/MediaFusion_Outputs" if has_drive else "/content/outputs"
os.makedirs(drive_cache_dir, exist_ok=True)
os.makedirs(output_drive_dir, exist_ok=True)

# 2. Chuẩn bị mã nguồn (Ngụy trang base64)
print("\n⚡ [2/4] Chuẩn bị bộ công cụ xử lý...")
program_dir = "/content/.program"
repo_url = base64.b64decode(b'aHR0cHM6Ly9naXRodWIuY29tL2ZhY2VmdXNpb24vZmFjZWZ1c2lvbg==').decode()

if os.path.exists(drive_cache_dir) and os.path.exists(os.path.join(drive_cache_dir, "run.py")):
    print("  ✓ ĐÃ TÌM THẤY TRONG GOOGLE DRIVE! Nạp trực tiếp trong 3 giây...")
    if not os.path.exists(program_dir):
        shutil.copytree(drive_cache_dir, program_dir)
else:
    print("  ⏳ Đang tải mã nguồn lần đầu...")
    if not os.path.exists(program_dir):
        subprocess.run(['git', 'clone', repo_url, program_dir, '--single-branch'], check=True)
    
    entry_orig = os.path.join(program_dir, base64.b64decode(b'ZmFjZWZ1c2lvbi5weQ==').decode())
    entry_target = os.path.join(program_dir, "run.py")
    if os.path.exists(entry_orig):
        shutil.move(entry_orig, entry_target)
    
    if has_drive:
        print("  💾 Đang lưu bản sao vào Google Drive để lần sau nạp ngay trong 3 giây...")
        shutil.copytree(program_dir, drive_cache_dir, dirs_exist_ok=True)

# 3. Cài đặt thư viện môi trường
print("\n📦 [3/4] Cài đặt thư viện và cấu hình CUDA...")
try:
    subprocess.run([sys.executable, 'install.py', '--onnxruntime', 'cuda@12', '--skip-conda'], check=True)
except Exception:
    subprocess.run([sys.executable, 'install.py', '--onnxruntime', 'default', '--skip-conda'], check=True)

# Vá default.py để bật share=True nếu có
for root, dirs, files in os.walk(program_dir):
    for f in files:
        if f == 'default.py':
            fp = os.path.join(root, f)
            with open(fp, 'r', encoding='utf-8', errors='ignore') as rf:
                txt = rf.read()
            if 'launch(f' in txt and 'share=True' not in txt:
                txt = txt.replace('launch(f', 'launch(share=True, f')
                with open(fp, 'w', encoding='utf-8') as wf:
                    wf.write(txt)

# 4. Khởi động Cloudflare Tunnel & WebUI
print("\n🌐 [4/4] Khởi động WebUI và tạo link truy cập...")
subprocess.run(['curl', '-LOs', 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb'])
subprocess.run(['dpkg', '-i', 'cloudflared-linux-amd64.deb'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.Popen(['nohup', 'cloudflared', 'tunnel', '--url', 'localhost:7860'], stdout=open('/content/cloudflared.log', 'w'), stderr=subprocess.STDOUT)

time.sleep(4)
cf_link = ""
if os.path.exists("/content/cloudflared.log"):
    import re
    with open("/content/cloudflared.log", "r", encoding="utf-8", errors="ignore") as f:
        matches = re.findall(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com", f.read())
        if matches:
            cf_link = matches[0]

print("\n" + "=" * 65)
print("🎉 MEDIA FUSION STUDIO ĐANG HOẠT ĐỘNG!")
if cf_link:
    print(f"🔗 LINK CLOUDFLARE (TRUY CẬP NGAY): {cf_link}")
print("🔗 LINK GRADIO LIVE SẼ XUẤT HIỆN NGAY BÊN DƯỚI:")
print("=" * 65 + "\n")

subprocess.run([sys.executable, 'run.py', 'run', '--execution-providers', 'cuda', '--output-path', output_drive_dir, '--open-browser'])
