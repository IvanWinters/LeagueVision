import os
import cv2
import numpy as np
import argparse

ICONS_FOLDER = 'champion_icons'
THRESHOLD = 0.8
ICON_RATIO = 25 / 280      # Reduced ratio for smaller icons
ICON_SEARCH_RATIO = .5    # Percentage of the champion icon to use in matchTemplate

def load_icons(folder, icon_size):
    icons = []
    for fname in os.listdir(folder):
        champ, ext = os.path.splitext(fname)
        if ext.lower() == '.png':
            path = os.path.join(folder, fname)
            icon = cv2.imread(path, cv2.IMREAD_COLOR)
            if icon is not None:
                # Resize to the icon size
                icon = cv2.resize(icon, (icon_size, icon_size), interpolation=cv2.INTER_CUBIC)

                # Crop the icon according to ICON_SEARCH_RATIO
                h, w = icon.shape[:2]
                crop_margin_w = int(w * ICON_SEARCH_RATIO / 2)
                crop_margin_h = int(h * ICON_SEARCH_RATIO / 2)
                icon = icon[crop_margin_h:h - crop_margin_h, crop_margin_w:w - crop_margin_w]

                icons.append((champ, icon))
    return icons

def process_video(input_video_path, icons_folder):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_video_path}")
        return

    # Read first frame to determine icon_size
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read the first frame.")
        return

    # Compute icon_size based on the frame's minimap size assumption
    # Here we assume the entire frame is the minimap for simplicity.
    h, w = frame.shape[:2]
    minimap_size = min(h, w)
    icon_size = int(minimap_size * ICON_RATIO)

    # Now load and adjust icons based on computed icon_size
    icons = load_icons(ICONS_FOLDER, icon_size)
    if not icons:
        print("No icons found. Place PNG icons in the champion_icons folder.")
        return

    print(f"Loaded {len(icons)} icons.")
    # Reset video to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected_frame = frame.copy()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for champion, icon in icons:
            gray_icon = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)

            # Ensure icon fits in the frame
            if gray_icon.shape[0] > gray_frame.shape[0] or gray_icon.shape[1] > gray_frame.shape[1]:
                continue

            # Run matchTemplate on the entire frame
            res = cv2.matchTemplate(gray_frame, gray_icon, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # Debug info
            print(f"{champion}: max_val = {max_val}")

            if max_val >= THRESHOLD:
                # Draw bounding box where match was found
                h_i, w_i = gray_icon.shape
                top_left = max_loc
                bottom_right = (top_left[0] + w_i, top_left[1] + h_i)
                cv2.rectangle(detected_frame, top_left, bottom_right, (0, 255, 255), 2)
                cv2.putText(detected_frame, f"{champion} ({max_val:.2f})",
                            (top_left[0], top_left[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 255), 1)

        cv2.imshow("Debug Detected", detected_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Minimal debug code for template matching with smaller icons.")
    parser.add_argument("input_video", help="Path to the input .mp4 file (already cropped to minimap).")
    args = parser.parse_args()

    process_video(args.input_video, ICONS_FOLDER)

if __name__ == "__main__":
    main()
