import os
import shutil
import time
import gradio as gr
from yt_dlp import YoutubeDL

os.makedirs("/content/Wav2Lip/results", exist_ok=True)
output_dir = "/content/drive/MyDrive/AI_Colab_Cache/Wav2Lip_Outputs" if os.path.exists("/content/drive/MyDrive") else "/content/cache/Wav2Lip_Outputs"
os.makedirs(output_dir, exist_ok=True)

def sync_lips(video_file, audio_file, youtube_url, pads_top, pads_bottom, pads_left, pads_right):
    target_video = "/content/input_video.mp4"
    if video_file is not None:
        shutil.copy(video_file, target_video)
    elif youtube_url and len(youtube_url.strip()) > 5:
        ydl_opts = {"overwrites": True, "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4", "outtmpl": target_video}
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download(youtube_url)
    else:
        return None, "❌ Vui lòng tải lên video hoặc dán link YouTube!"
    
    if audio_file is None:
        return None, "❌ Vui lòng tải lên file âm thanh tiếng Việt!"
    
    output_path = "/content/Wav2Lip/results/result_voice.mp4"
    cmd = f"python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face '{target_video}' --audio '{audio_file}' --pads {pads_top} {pads_bottom} {pads_left} {pads_right} --outfile '{output_path}'"
    os.system(cmd)
    
    if os.path.exists(output_path):
        ts_name = f"lipsync_{int(time.time())}.mp4"
        shutil.copy(output_path, f"{output_dir}/{ts_name}")
        return output_path, f"🎉 Khớp khẩu hình thành công! Đã tự động lưu ({ts_name})."
    else:
        return None, "❌ Có lỗi trong quá trình xử lý. Hãy kiểm tra lại định dạng video/âm thanh!"

with gr.Blocks(title="Wav2Lip Cloud Studio") as demo:
    gr.Markdown("## 🎬 Wav2Lip AI - Khớp Khẩu Hình Siêu Tốc (Tesla T4 Cloud GPU)")
    gr.Markdown("Tải lên video gốc của bạn và file âm thanh lồng tiếng tiếng Việt để AI tự động đồng bộ cử động môi theo nhịp nói.")
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="1. Tải lên Video gốc (MP4)")
            yt_input = gr.Textbox(label="Hoặc dán Link YouTube nếu không tải video lên", placeholder="https://youtu.be/...")
            audio_input = gr.Audio(label="2. Tải lên File Âm thanh Tiếng Việt (MP3 / WAV)", type="filepath")
            with gr.Accordion("Tùy chỉnh đệm viền môi (Pads) - Mặc định chuẩn", open=False):
                p_top = gr.Slider(0, 20, value=0, step=1, label="Pads Top")
                p_bottom = gr.Slider(0, 20, value=10, step=1, label="Pads Bottom (Đệm cằm)")
                p_left = gr.Slider(0, 20, value=0, step=1, label="Pads Left")
                p_right = gr.Slider(0, 20, value=0, step=1, label="Pads Right")
            btn_run = gr.Button("🚀 Bắt Đầu Khớp Khẩu Hình (Generate Lip-Sync)", variant="primary")
        with gr.Column():
            video_output = gr.Video(label="Video Kết Quả (Đã Khớp Môi)")
            status_msg = gr.Textbox(label="Trạng thái xử lý", interactive=False)
    btn_run.click(sync_lips, inputs=[video_input, audio_input, yt_input, p_top, p_bottom, p_left, p_right], outputs=[video_output, status_msg])

if __name__ == '__main__':
    demo.queue().launch(share=True)
