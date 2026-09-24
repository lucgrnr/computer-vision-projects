# README

This repository contains the code and documentation for an individual assignment in a computer vision course. The assignment involves processing a video to detect and manipulate objects using basic computer vision techniques, including grayscale conversion, blurring, edge detection, circle detection, and template matching.

## What It Covers

**Basic Image Processing:**
- Grayscale conversion.
- Gaussian and bilateral blurring.
- Sobel edge detection in horizontal and vertical directions.

**Color-Based Object Isolation:**
- Object isolation using RGB and HSV color spaces.
- Mask refinement using morphological operations.

**Circle Detection:**
- Hough circle transform for detecting circular objects with various parameters.

**Template Matching:**
- Template matching using normalized cross-correlation to locate objects in the video.

**Effects Driven by Detections:**
- Manipulating objects based on their detected positions, such as changing their color or duplicating them at different locations.

## Files

- `code_assign1_lg.ipynb`: The main Jupyter notebook containing the code for video processing.
- `video.mp4`: The source video file.
- `processed_video.mp4`: The output video after processing.
- `compressed_video.mp4`: The compressed output video.
- `template_body.png`: Template image for body detection.
- `template_eyes.png`: Template image for eyes detection.

## Running the Code

1. **Install Required Packages:**
   Ensure that you have the necessary Python packages installed. Open the notebook and run the installation cell at the beginning.

   ```python
   #!pip install -q --upgrade pip
   #!pip install opencv-python numpy moviepy
   ```

2. **Mount Google Drive:**
   If you are using Google Colab, mount your Google Drive to access the video files.

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Set File Paths:**
   Define the paths to the input and output video files.

   ```python
   folder = '/content/drive/MyDrive/Cours KU/S2/Computer Vision/Project/Individual assignment 1/'
   original_video = folder + 'PXL_20260219_212014081.mp4'
   downsized_video = folder + 'downsized_video.mp4'
   processed_video = folder + 'processed_video.mp4'
   compressed_video = folder + 'compressed_video.mp4'
   ```

4. **Run the Video Processing:**
   Execute the main function to process the video.

   ```python
   main(downsized_video, processed_video)
   ```

5. **Compress the Output Video:**
   Use MoviePy to compress the video to reduce its size.

   ```python
   from moviepy.editor import VideoFileClip

   clip = VideoFileClip(processed_video)
   clip.write_videofile(compressed_video, bitrate="1500k", codec="libx264", audio=False)
   ```

## Notes

- **Debugging:**
  Set the `SHOW_FRAME_AT` variable to a timestamp in milliseconds to display a specific frame for debugging purposes.
  
  ```python
  SHOW_FRAME_AT = 24000
  ```

- **Template Matching:**
  Templates are created from the first frame of the video and used to detect objects in subsequent frames. The coordinates for the body template are `(275, 100, 250, 375)` and for the eyes template are `(385, 200, 125, 60)`. Adjust these coordinates as needed.

  ```python
  template_body_loc = 275, 100, 250, 375
  template_eyes_loc = 385, 200, 125, 60
  ```

This README provides a comprehensive guide to understanding and running the video processing code.