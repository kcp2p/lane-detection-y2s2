"""
Main ANPR system class to coordinate detection and recognition.
"""

import cv2
import time
import os
import glob
import numpy as np
from plate_detector import LicensePlateDetector
from plate_recognizer import LicensePlateRecognizer
from lane_detector import LaneDetector


class ANPRSystem:
    """Main ANPR system class to coordinate detection and recognition."""

    def __init__(self, debug_mode=False, lane_type=None):
        """Initialize the ANPR system.

        Args:
            debug_mode (bool): Whether to show intermediate processing steps
            lane_type (str): Type of lane detection to use (None, "straight", "curved")
        """
        self.detector = LicensePlateDetector(debug_mode)
        self.recognizer = LicensePlateRecognizer()
        self.lane_detector = LaneDetector(debug_mode, lane_type) if lane_type else None
        self.debug_mode = debug_mode
        self.lane_type = lane_type

    def process_image(self, image_path):
        """Process a single image to detect and recognize license plates.

        Args:
            image_path: Path to the input image

        Returns:
            list: List of recognized license plate texts
        """
        start_time = time.time()

        input_image = cv2.imread(image_path)
        if input_image is None:
            print(f"Error: Could not read image {image_path}")
            return []

        if self.lane_detector:
            print(f"Detecting {self.lane_type} lanes...")
            input_image = self.lane_detector.detect_lanes(input_image)

            if self.debug_mode:
                cv2.imshow("Lane Detection Result", input_image)
                print("Showing lane detection result - Press any key to continue...")
                cv2.waitKey(0)

        detection, crops = self.detector.detect(input_image)

        if self.debug_mode:
            cv2.imshow("Detected License Plates", detection)
            print("Showing detected license plates - Press any key to continue...")
            cv2.waitKey(0)

        results = []
        for i, crop in enumerate(crops):

            processed = self.detector.process_plate(crop)

            text = self.recognizer.recognize(processed)

            results.append(text)
            print(f"Plate {i+1}: {text}")

            if self.debug_mode:
                result_display = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                cv2.putText(
                    result_display,
                    text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(f"Recognized Plate {i+1}", result_display)
                print(f"Showing recognized plate {i+1} - Press any key to continue...")
                cv2.waitKey(0)

        cv2.imshow("Final Detection Result", detection)
        print("Showing final detection result - Press any key to continue...")
        cv2.waitKey(0)

        end_time = time.time()
        print(f"Processing time: {end_time - start_time:.2f} seconds")

        cv2.destroyAllWindows()

        return results

    def process_video(self, video_path):
        """Process a video to detect license plates in each frame.

        Args:
            video_path: Path to the input video
        """
        while True:

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print("Error: Could not open video file")
                return

            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            process_every_n_frames = max(1, int(fps / 2))

            cv2.namedWindow("ANPR Video Processing", cv2.WINDOW_NORMAL)

            last_detected_plates = []
            last_detected_rois = []
            last_processed_plates = []

            while cap.isOpened():

                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                if self.lane_detector:
                    frame = self.lane_detector.detect_lanes(frame)

                if frame_count % process_every_n_frames == 0:

                    processed_frame, crops = self.detector.detect(frame)

                    if crops:
                        current_plates = []
                        current_rois = []
                        current_processed_plates = []

                        print("\nLicense Plates Detected (Video):")
                        print("-----------------------")

                        contours = self._find_plate_contours(processed_frame)

                        for i, crop in enumerate(crops):

                            processed_crop = self.detector.process_plate(crop)

                            processed_display = cv2.cvtColor(
                                processed_crop, cv2.COLOR_GRAY2BGR
                            )

                            text = self.recognizer.recognize(processed_crop)

                            cv2.putText(
                                processed_display,
                                text,
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2,
                                cv2.LINE_AA,
                            )

                            current_plates.append(text)
                            current_processed_plates.append(processed_display)

                            if i < len(contours):
                                current_rois.append(contours[i])

                            print(f"Plate {i+1}: {text}")

                        print("-----------------------")

                        if current_plates:
                            last_detected_plates = current_plates
                            last_detected_rois = current_rois
                            last_processed_plates = current_processed_plates

                    frame = processed_frame

                display_frame = self._create_display_with_overlays(
                    frame,
                    last_detected_plates,
                    last_detected_rois,
                    last_processed_plates,
                )

                cv2.imshow("ANPR Video Processing", display_frame)

                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            cap.release()

    def _create_display_with_overlays(self, frame, texts, rois, processed_plates):
        """Create a display frame with overlays for processed plates.

        Args:
            frame: Original video frame
            texts: List of recognized license plate texts
            rois: List of license plate ROI contours
            processed_plates: List of processed license plate images

        Returns:
            Image with license plate overlay panels
        """

        display = frame.copy()

        panel_width = 240
        panel_height = 120
        margin = 400
        start_y = 120

        cv2.putText(
            display,
            "Press 'Q' to exit",
            (650, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        for i, roi in enumerate(rois):
            cv2.drawContours(display, [roi], 0, (0, 0, 255), 2)

            M = cv2.moments(roi)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                cv2.putText(
                    display,
                    f"{i+1}",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

        for i, (text, plate_img) in enumerate(zip(texts, processed_plates)):

            panel_x = display.shape[1] - panel_width - margin
            panel_y = start_y + i * (panel_height + margin)

            if panel_y + panel_height > display.shape[0]:
                break

            cv2.rectangle(
                display,
                (panel_x - 5, panel_y - 5),
                (panel_x + panel_width + 5, panel_y + panel_height + 5),
                (0, 0, 0),
                -1,
            )
            cv2.rectangle(
                display,
                (panel_x - 5, panel_y - 5),
                (panel_x + panel_width + 5, panel_y + panel_height + 5),
                (255, 255, 255),
                2,
            )

            cv2.putText(
                display,
                f"Plate {i+1}: {text}",
                (panel_x, panel_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            resized_plate = cv2.resize(plate_img, (panel_width, panel_height))

            display[
                panel_y : panel_y + panel_height, panel_x : panel_x + panel_width
            ] = resized_plate

        return display

    def _find_plate_contours(self, frame):
        """Find license plate contours in the processed frame.

        This is a helper method to extract the ROI contours of license plates
        that have been marked by the detector.

        Args:
            frame: Processed frame with license plate annotations

        Returns:
            list: List of contours for license plates
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        red_mask = cv2.inRange(frame, (0, 0, 200), (50, 50, 255))

        contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        min_area = 500
        plate_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        return plate_contours

    def process_directory(self, directory_path):
        """Process all images in a directory.

        Args:
            directory_path: Path to the directory containing images

        Returns:
            dict: Dictionary mapping filenames to recognized plate texts
        """

        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        image_files = []

        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(directory_path, ext)))

        if not image_files:
            print(f"No image files found in {directory_path}")
            return {}

        results = {}
        for image_file in image_files:
            print(f"\nProcessing {os.path.basename(image_file)}...")
            plate_texts = self.process_image(image_file)
            results[os.path.basename(image_file)] = plate_texts

        return results
