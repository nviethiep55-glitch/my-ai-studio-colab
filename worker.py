import os
import sys
import time
import queue
import shutil
import tempfile
import threading
import subprocess
import glob
import ctypes
from typing import Optional

# 1. Cấu hình biến môi trường LD_LIBRARY_PATH cho CUDA 12
try:
    import site
    for sp in site.getsitepackages():
        nv = os.path.join(sp, 'nvidia')
        if os.path.exists(nv):
            for sub in os.listdir(nv):
                lib_p = os.path.join(nv, sub, 'lib')
                if os.path.isdir(lib_p):
                    os.environ["LD_LIBRARY_PATH"] = f"{lib_p}:{os.environ.get('LD_LIBRARY_PATH', '')}"
except Exception:
    pass

# 2. Cực kỳ quan trọng: Nạp trước các thư viện CUDA 12 / cuDNN vào Global Symbol Table (RTLD_GLOBAL)
cuda_so_candidates = [
    "libcudart.so", "libcudart.so.12",
    "libnvrtc.so", "libnvrtc.so.12",
    "libcublasLt.so", "libcublasLt.so.12",
    "libcublas.so", "libcublas.so.12",
    "libcufft.so", "libcufft.so.12",
    "libcudnn.so", "libcudnn.so.9", "libcudnn.so.8"
]
search_dirs = ["/usr/lib", "/usr/local/cuda/lib64", "/usr/local/cuda-12/lib64"]
try:
    import site
    for sp in site.getsitepackages():
        search_dirs.extend(glob.glob(f"{sp}/nvidia/*/lib"))
except Exception:
    pass

for lib in cuda_so_candidates:
    try:
        ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        continue
    except Exception:
        pass
    for d in search_dirs:
        fp = os.path.join(d, lib)
        if os.path.exists(fp):
            try:
                ctypes.CDLL(fp, mode=ctypes.RTLD_GLOBAL)
                break
            except Exception:
                pass

# 3. Preload torch for CUDA libraries & warmup GPU context
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.init()
        _ = torch.zeros(1).cuda()
except Exception:
    pass

try:
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
except Exception:
    pass

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="My AI Studio - Colab GPU Face Swap Worker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global progress tracker
current_progress = {
    "current_frame": 0,
    "total_frames": 0,
    "fps": 0.0,
    "status": "idle"
}

# Cache models
face_app = None
swapper = None
enhancer = None
models_dir = os.environ.get("MODELS_DIR", "/content/models")
os.makedirs(models_dir, exist_ok=True)

def init_models():
    global face_app, swapper
    if face_app is not None and swapper is not None:
        return

    print("⚡ [Colab Worker] Khởi tạo mô hình AI trên CUDA GPU...", flush=True)
    cuda_opts = {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 14 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'DEFAULT',
        'do_copy_in_default_stream': True
    }
    providers = [('CUDAExecutionProvider', cuda_opts), 'CPUExecutionProvider']
    face_app = FaceAnalysis(name='buffalo_l', providers=providers)
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    swapper_path = os.path.join(models_dir, "inswapper_128.onnx")
    if not os.path.exists(swapper_path):
        print("  ⏳ Đang tải trọng số inswapper_128.onnx (528MB)...", flush=True)
        url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
        subprocess.run(["wget", "-c", url, "-O", swapper_path], check=True)

    swapper = insightface.model_zoo.get_model(swapper_path, download=False, providers=providers)
    
    # Verify active provider
    try:
        active_list = swapper.session.get_providers()
        print(f"🚀 [Colab Worker] ONNX Runtime Providers: {active_list}", flush=True)
        if 'CUDAExecutionProvider' in active_list:
            print("🔥 TURBO GPU KÍCH HOẠT THÀNH CÔNG! Tốc độ dự kiến ~35-45 FPS.", flush=True)
        else:
            print("⚠️ CẢNH BÁO: Đang chạy trên CPU do thiếu cuDNN! Hãy kiểm tra cài đặt nvidia-cudnn-cu12.", flush=True)
    except Exception:
        pass

    print("✅ [Colab Worker] Mô hình đã sẵn sàng!", flush=True)

def compute_similarity(emb1, emb2):
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

@app.get("/health")
def health_check():
    gpu_name = "CPU"
    vram_str = "N/A"
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            mem_free, mem_total = torch.cuda.mem_get_info()
            vram_str = f"{(mem_total - mem_free)/(1024**3):.1f}GB / {mem_total/(1024**3):.1f}GB"
    except Exception:
        pass

    active_provider = "CPUExecutionProvider"
    if swapper is not None:
        try:
            active_provider = swapper.session.get_providers()[0]
        except Exception:
            pass

    return {
        "status": "ok",
        "gpu": gpu_name,
        "vram": vram_str,
        "provider": active_provider,
        "model_ready": (swapper is not None and face_app is not None),
        "webui_url": os.environ.get("WEBUI_URL", ""),
        "message": f"Colab GPU Worker sẵn sàng ({gpu_name} - Provider: {active_provider})!"
    }

@app.get("/progress")
def get_progress():
    return current_progress

@app.post("/swap")
async def process_face_swap(
    source_image: UploadFile = File(...),
    target_video: UploadFile = File(...),
    target_ref_image: Optional[UploadFile] = File(None),
    target_face_image: Optional[UploadFile] = File(None),
    has_target_ref: str = Form("false"),
    similarity_threshold: float = Form(0.40),
    use_gfpgan: str = Form("false"),
    turbo_threads: int = Form(2)
):
    init_models()

    tmp_dir = tempfile.mkdtemp(prefix="faceswap_")
    try:
        # Save uploaded source face image
        src_path = os.path.join(tmp_dir, "source_face.jpg")
        with open(src_path, "wb") as f:
            shutil.copyfileobj(source_image.file, f)

        src_bgr = cv2.imread(src_path)
        if src_bgr is None:
            raise HTTPException(status_code=400, detail="Không thể đọc ảnh mặt mới (source_image)")

        src_faces = face_app.get(src_bgr)
        if len(src_faces) == 0:
            raise HTTPException(status_code=400, detail="Không tìm thấy khuôn mặt nào trong ảnh mặt mới")

        source_face = max(src_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))

        # Handle target reference face (selective swap for multi-character video)
        target_ref_face = None
        tgt_upload = target_ref_image or target_face_image
        has_ref = (has_target_ref.lower() in ("true", "1") or target_face_image is not None) and tgt_upload is not None
        if has_ref and tgt_upload is not None:
            ref_path = os.path.join(tmp_dir, "ref_target.jpg")
            with open(ref_path, "wb") as f:
                shutil.copyfileobj(tgt_upload.file, f)
            ref_bgr = cv2.imread(ref_path)
            if ref_bgr is not None:
                r_faces = face_app.get(ref_bgr)
                if len(r_faces) > 0:
                    target_ref_face = max(r_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))

        # Save uploaded video
        vid_in_path = os.path.join(tmp_dir, "input_video.mp4")
        with open(vid_in_path, "wb") as f:
            shutil.copyfileobj(target_video.file, f)

        cap = cv2.VideoCapture(vid_in_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Không thể mở file video gốc")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        temp_raw_out = os.path.join(tmp_dir, "raw_swapped.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(temp_raw_out, fourcc, fps, (width, height))

        # Reset progress
        current_progress["status"] = "processing"
        current_progress["current_frame"] = 0
        current_progress["total_frames"] = total_frames
        current_progress["fps"] = 0.0

        # Multi-Threaded Prefetch Buffer
        frame_queue = queue.Queue(maxsize=32)
        write_queue = queue.Queue(maxsize=32)
        read_done = threading.Event()
        swap_done = threading.Event()

        def reader_worker():
            f_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_queue.put((f_idx, frame))
                f_idx += 1
            cap.release()
            read_done.set()

        def writer_worker():
            while not (swap_done.is_set() and write_queue.empty()):
                try:
                    f = write_queue.get(timeout=0.2)
                    out_writer.write(f)
                    write_queue.task_done()
                except queue.Empty:
                    continue
            out_writer.release()

        threading.Thread(target=reader_worker, daemon=True).start()
        threading.Thread(target=writer_worker, daemon=True).start()

        start_time = time.time()
        processed = 0

        while not (read_done.is_set() and frame_queue.empty()):
            try:
                frame_idx, frame = frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                faces = face_app.get(frame)
                if len(faces) > 0:
                    if target_ref_face is not None:
                        # Match target face by cosine similarity
                        for f in faces:
                            sim = compute_similarity(f.embedding, target_ref_face.embedding)
                            if sim >= similarity_threshold:
                                frame = swapper.get(frame, f, source_face, paste_back=True)
                    else:
                        # Swap largest face in frame
                        largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                        frame = swapper.get(frame, largest, source_face, paste_back=True)
            except Exception as e:
                pass

            write_queue.put(frame)
            frame_queue.task_done()
            processed += 1

            if processed % 5 == 0 or processed == total_frames:
                elapsed = max(0.001, time.time() - start_time)
                cur_fps = round(processed / elapsed, 1)
                current_progress["current_frame"] = processed
                current_progress["fps"] = cur_fps

        swap_done.set()
        while not write_queue.empty():
            time.sleep(0.05)

        # Merge original audio back using FFmpeg
        final_out = os.path.join(tmp_dir, "final_swapped.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_raw_out,
            "-i", vid_in_path,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            final_out
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

        if not os.path.exists(final_out) or os.path.getsize(final_out) == 0:
            final_out = temp_raw_out

        current_progress["status"] = "completed"
        current_progress["current_frame"] = total_frames

        return FileResponse(
            final_out,
            media_type="video/mp4",
            filename=f"swapped_{target_video.filename}"
        )

    except Exception as e:
        current_progress["status"] = "error"
        raise HTTPException(status_code=500, detail=str(e))

def run_server(port: int = 8000):
    init_models()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    run_server(port)
