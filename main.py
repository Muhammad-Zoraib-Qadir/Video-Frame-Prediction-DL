import streamlit as st
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
import os
import subprocess
from models import VideoPredictor, PredRNN, VideoTransformer

try:
   subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
except subprocess.CalledProcessError:
   st.error("FFmpeg is not installed or not found in PATH. Please install FFmpeg first.")
   st.stop()

def load_model(model_name, model_path, num_input_frames, num_pred_frames, mode='rgb'):
   input_channels = 1 if mode == 'grayscale' else 3
   
   if model_name == "VideoPredictor":
       model = VideoPredictor(
           nf=20,
           input_frames=num_input_frames,
           output_frames=num_pred_frames,
           input_channels=input_channels
       )
   elif model_name == "PredRNN":
       model = PredRNN(
           input_channels=input_channels,
           num_hidden=[32, 32, 32],
           input_frames=num_input_frames,
           output_frames=num_pred_frames
       )
   else:  # VideoTransformer
       model = VideoTransformer(
           input_channels=3,
           d_model=256,
           nhead=8,
           num_layers=6,
           dim_feedforward=1024,
           input_frames=num_input_frames,
           output_frames=num_pred_frames
       )
   
   checkpoint = torch.load(model_path, map_location='cpu')
   model.load_state_dict(checkpoint['state_dict'], strict=True)
   model.eval()
   model.to('cuda' if torch.cuda.is_available() else 'cpu')
   return model

def process_video(model, video_file, num_input_frames=10, num_pred_frames=5, mode='rgb'):
   # Create predicted_videos directory if it doesn't exist
   os.makedirs('predicted_videos', exist_ok=True)
   
   # Save uploaded video temporarily
   temp_path = os.path.join('predicted_videos', 'temp_input.mp4')
   with open(temp_path, 'wb') as f:
       f.write(video_file.read())

   def load_video_frames(video_path, frame_size=(64, 64)):
       cap = cv2.VideoCapture(video_path)
       frames = []
       fps = cap.get(cv2.CAP_PROP_FPS)
       while True:
           ret, frame = cap.read()
           if not ret:
               break
           if mode == 'grayscale':
               frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
           else:
               frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           resized = cv2.resize(frame, frame_size)
           frames.append(resized)
       cap.release()
       return np.array(frames), fps

   def preprocess_frames(frames):
       if mode == 'grayscale':
           frames = frames[..., np.newaxis]
       frames = torch.from_numpy(frames).float() / 255.0
       if mode == 'grayscale':
           frames = frames.permute(0, 3, 1, 2)
       else:
           frames = frames.permute(0, 3, 1, 2)
       return frames.unsqueeze(0)

   # Load all frames
   all_frames, fps = load_video_frames(temp_path)
   total_frames = len(all_frames)
   os.remove(temp_path) 

   # Generate output paths
   temp_avi_path = os.path.join('predicted_videos', 'temp_output.avi')
   output_path = os.path.join('predicted_videos', os.path.splitext(video_file.name)[0] + '_predicted.mp4')
   
   # Create video writer with MJPG codec for temporary AVI
   fourcc = cv2.VideoWriter_fourcc(*'MJPG')
   frame_size = all_frames[0].shape[:2]
   out = cv2.VideoWriter(temp_avi_path, fourcc, fps, (frame_size[1], frame_size[0]), True)
   
   # Process video in chunks
   start_idx = 0
   first_iteration = True
   first_metrics = None
   first_visualization_data = None

   while start_idx + num_input_frames < total_frames:
       # Get input frames
       input_frames = all_frames[start_idx:start_idx + num_input_frames]
       
       # Get ground truth
       end_idx = min(start_idx + num_input_frames + num_pred_frames, total_frames)
       ground_truth = all_frames[start_idx + num_input_frames:end_idx]
       
       if len(ground_truth) < num_pred_frames:
           break
           
       # Preprocess frames
       input_tensor = preprocess_frames(input_frames)
       ground_truth_tensor = preprocess_frames(ground_truth)
       
       # Make prediction
       with torch.no_grad():
           input_tensor = input_tensor.to(model.device)
           predicted_frames = model(input_tensor)
       
       pred_frames = predicted_frames[0].cpu().numpy().transpose(0, 2, 3, 1)
       
       # Calculate metrics
       mse = F.mse_loss(predicted_frames, ground_truth_tensor.to(model.device))
       ssim_val = ssim(predicted_frames, ground_truth_tensor.to(model.device), data_range=1.0)
       
       # Store first iteration data
       if first_iteration:
           first_metrics = (mse.item(), ssim_val.item())
           first_visualization_data = (input_frames, ground_truth, pred_frames)
           first_iteration = False
       
       # Write original frames to video in grayscale
       for frame in input_frames:
           if mode == 'grayscale':
               frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
           else:
               frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
               frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
           out.write((frame_bgr * 255).astype(np.uint8))
       
       # Write predicted frames to video in color
       for frame in pred_frames:
           if mode == 'grayscale':
               frame = frame.squeeze()
               colored_frame = np.stack([frame * 0.5, frame, frame * 0.5], axis=-1)  # Green tint
               frame_bgr = cv2.cvtColor((colored_frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
           else:
               frame_bgr = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
           out.write(frame_bgr)
       
       # Move to next chunk
       start_idx += num_input_frames + num_pred_frames
   
   out.release()

   # Convert AVI to MP4 using FFmpeg
   try:
       command = [
           'ffmpeg',
           '-i', temp_avi_path,  # Input file
           '-c:v', 'libx264',    # Video codec
           '-preset', 'medium',   # Encoding speed preset
           '-crf', '23',         # Quality (lower is better, 18-28 is good)
           '-y',                 # Overwrite output file
           output_path           # Output file
       ]
       
       subprocess.run(command, check=True, capture_output=True)
       os.remove(temp_avi_path)
       
   except subprocess.CalledProcessError as e:
       st.error(f"Error converting video: {e.stderr.decode()}")
       if os.path.exists(temp_avi_path):
           os.remove(temp_avi_path)
       return None, None, None
   except Exception as e:
       st.error(f"Error: {str(e)}")
       if os.path.exists(temp_avi_path):
           os.remove(temp_avi_path)
       return None, None, None

   return first_visualization_data, first_metrics, output_path

def main():
   st.title("Video Prediction App")
   
   model_options = {
       "Convolutional LSTM": "models/convlstm-epoch=19-val_combined_loss=0.19.ckpt",
       "PredRNN": "models/predrnn-epoch=09-val_mse=0.2703-val_ssim=0.1524.ckpt",
       "Transformer-based model": "models/transformer-epoch=04-val_mse=0.3555.ckpt"
   }
   
   selected_model = st.selectbox("Select Model", list(model_options.keys()))
   uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi'])
   
   if uploaded_file is not None:
       if st.button("Predict"):
           with st.spinner("Processing video..."):
               model = load_model(selected_model, model_options[selected_model], 10, 5)
               
               visualization_data, metrics, output_path = process_video(model, uploaded_file)
               
               if visualization_data is not None:
                   input_frames, ground_truth, pred_frames = visualization_data
                   mse, ssim_val = metrics
                   
                   # Display metrics
                   col1, col2 = st.columns(2)
                   col1.metric("MSE", f"{mse:.4f}")
                   col2.metric("SSIM", f"{ssim_val:.4f}")
                   
                   # Display frames
                   st.subheader("Input Frames")
                   fig1, ax1 = plt.subplots(1, len(input_frames), figsize=(15, 3))
                   for i, frame in enumerate(input_frames):
                       ax1[i].imshow(frame)
                       ax1[i].axis('off')
                       ax1[i].set_title(f'Input {i+1}')
                   st.pyplot(fig1)
                   
                   st.subheader("Ground Truth vs Predictions")
                   fig2, ax2 = plt.subplots(2, len(ground_truth), figsize=(15, 6))
                   for i in range(len(ground_truth)):
                       ax2[0, i].imshow(ground_truth[i])
                       ax2[0, i].axis('off')
                       ax2[0, i].set_title(f'Ground Truth {i+1}')
                       
                       ax2[1, i].imshow(pred_frames[i])
                       ax2[1, i].axis('off')
                       ax2[1, i].set_title(f'Predicted {i+1}')
                   st.pyplot(fig2)
                   
                   # Display predicted video
                   st.subheader("Predicted Video")
                   video_file = open(output_path, 'rb')
                   video_bytes = video_file.read()
                   st.video(video_bytes)
                   video_file.close()
                   
                   st.success(f"Video saved to: {output_path}")

if __name__ == "__main__":
   main()