import cv2
import argparse
import os

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Crop the minimap from a League of Legends clip.")
    parser.add_argument("input_video", help="Path to the input video file, e.g. data/clip1/input.mp4")
    parser.add_argument("output_dir", help="Directory where the output file 'minimap.mp4' will be saved, e.g. data/")
    args = parser.parse_args()

    input_video_path = args.input_video
    output_dir = args.output_dir

    # Ensure output directory exists
    if not os.path.isdir(output_dir):
        raise NotADirectoryError(f"The specified output directory does not exist: {output_dir}")

    # Set the output video path to minimap.mp4 in the given directory
    output_video_path = os.path.join(output_dir, "minimap.mp4")

    # Set up the ratio as given
    minimap_ratio = 800 / 1080  # ratio used to determine minimap size from video frame height

    # Open the video capture
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open the video from {input_video_path}")

    # Read the first frame to determine dimensions and cropping parameters
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Couldn't read the first frame of the video.")

    height, width, _ = frame.shape

    # Calculate the minimap size
    minimap_x = int(height * minimap_ratio)  # scaled dimension
    minimap_size = height - minimap_x

    # The cropped area will always be from the bottom-right corner
    crop_x_start = width - minimap_size
    crop_y_start = height - minimap_size

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (minimap_size, minimap_size))

    # Reset back to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # No more frames

        # Crop the frame
        minimap_frame = frame[crop_y_start:crop_y_start+minimap_size, crop_x_start:crop_x_start+minimap_size]

        # Write the cropped frame to the output video
        out.write(minimap_frame)

    # Release all resources
    cap.release()
    out.release()

    print(f"Cropped minimap video has been saved as {output_video_path}")

if __name__ == "__main__":
    main()
