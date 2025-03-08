"""
Lane detection using traditional computer vision techniques.
Supports curved lane detection.
"""

import cv2
import numpy as np


class LaneDetector:
    """Lane detection using traditional computer vision techniques."""

    def __init__(self, debug_mode=False, lane_type="curved"):
        """Initialize the lane detector.

        Args:
            debug_mode (bool): Whether to show intermediate processing steps
            lane_type (str): Type of lane detection to use (only "curved" supported)
        """
        self.debug_mode = debug_mode
        if lane_type != "curved":
            print("Only curved lane detection is supported, using curved")
        self.lane_type = "curved"

    def detect_lanes(self, image):
        """Detect lanes in an image.

        Args:
            image: Input BGR image

        Returns:
            Image with lane overlays
        """

        frame = image.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

        processed_img = self._preprocessing_curved(frame)

        height, width = processed_img.shape
        polygon = [
            (int(width * 0.15), int(height * 0.94)),
            (int(width * 0.45), int(height * 0.62)),
            (int(width * 0.58), int(height * 0.62)),
            (int(0.95 * width), int(0.94 * height)),
        ]

        masked_img = self._region_of_interest(processed_img, polygon)

        source_points = np.float32(
            [
                [int(width * 0.49), int(height * 0.62)],
                [int(width * 0.58), int(height * 0.62)],
                [int(width * 0.15), int(height * 0.94)],
                [int(0.95 * width), int(0.94 * height)],
            ]
        )
        destination_points = np.float32([[0, 0], [400, 0], [0, 960], [400, 960]])
        warped_img_size = (400, 960)
        warped_img_shape = (960, 400)

        warped_img = self._warp(
            masked_img, source_points, destination_points, warped_img_size
        )

        kernel = np.ones((11, 11), np.uint8)
        opening = cv2.morphologyEx(warped_img, cv2.MORPH_CLOSE, kernel)

        left_fit, right_fit = self._fit_curve(opening)
        pts_left, pts_right = self._find_points(warped_img_shape, left_fit, right_fit)

        fill_curves = self._fill_curves(warped_img_shape, pts_left, pts_right)

        unwarped_fill_curves = self._unwarp(
            fill_curves, source_points, destination_points, (width, height)
        )
        result_with_lane = cv2.addWeighted(frame, 1, unwarped_fill_curves, 1, 0)

        left_radius, right_radius, avg_radius = self._radius_of_curvature(
            warped_img, left_fit, right_fit
        )

        window1 = result_with_lane
        window2 = self._one_to_three_channel(thresh)
        window3 = self._one_to_three_channel(warped_img)
        window4 = self._draw_curves(warped_img, pts_left, pts_right)
        window5 = self._information_window(left_radius, right_radius, avg_radius)

        combined_result = self._concatenate(window1, window2, window3, window4, window5)

        final_result = self._add_turn_info(combined_result, avg_radius)

        if self.debug_mode:
            self._debug_show("1. Preprocessed Image", processed_img)
            self._debug_show("2. Region of Interest", masked_img)
            self._debug_show("3. Warped Image", warped_img)
            self._debug_show("4. Enhanced Lane Markings", opening)
            self._debug_show("5. Curve Fitting", window4)

        return final_result

    def _preprocessing_curved(self, img):
        """Preprocess image for curved lane detection."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gblur = cv2.GaussianBlur(gray, (5, 5), 0)
        white_mask = cv2.threshold(gblur, 200, 255, cv2.THRESH_BINARY)[1]
        lower_yellow = np.array([0, 100, 100])
        upper_yellow = np.array([210, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        return mask

    def _region_of_interest(self, img, polygon):
        """Apply region of interest mask to an image."""
        mask = np.zeros_like(img)

        if len(img.shape) > 2:

            channel_count = img.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:

            ignore_mask_color = 255

        cv2.fillPoly(mask, np.array([polygon], np.int32), ignore_mask_color)

        masked_img = cv2.bitwise_and(img, mask)

        return masked_img

    def _warp(self, img, source_points, destination_points, destn_size):
        """Warp image to bird's eye view."""
        matrix = cv2.getPerspectiveTransform(source_points, destination_points)
        warped_img = cv2.warpPerspective(img, matrix, destn_size)
        return warped_img

    def _unwarp(self, img, source_points, destination_points, source_size):
        """Unwarp image back to original perspective."""
        matrix = cv2.getPerspectiveTransform(destination_points, source_points)
        unwarped_img = cv2.warpPerspective(img, matrix, source_size)
        return unwarped_img

    def _fit_curve(self, img):
        """Fit curve to lane lines in warped image."""

        histogram = np.sum(img[img.shape[0] // 2 :, :], axis=0)
        midpoint = int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 50
        margin = 100
        minpix = 50
        window_height = int(img.shape[0] / nwindows)

        nonzero = img.nonzero()
        y = nonzero[0]
        x = nonzero[1]

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_indices = []
        right_lane_indices = []

        for window in range(nwindows):
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_indices = (
                (y >= win_y_low)
                & (y < win_y_high)
                & (x >= win_xleft_low)
                & (x < win_xleft_high)
            ).nonzero()[0]
            good_right_indices = (
                (y >= win_y_low)
                & (y < win_y_high)
                & (x >= win_xright_low)
                & (x < win_xright_high)
            ).nonzero()[0]

            left_lane_indices.append(good_left_indices)
            right_lane_indices.append(good_right_indices)

            if len(good_left_indices) > minpix:
                leftx_current = int(np.mean(x[good_left_indices]))
            if len(good_right_indices) > minpix:
                rightx_current = int(np.mean(x[good_right_indices]))

        try:
            left_lane_indices = np.concatenate(left_lane_indices)
            right_lane_indices = np.concatenate(right_lane_indices)
        except:

            return np.array([0, 0, 0]), np.array([0, 0, 0])

        leftx = x[left_lane_indices]
        lefty = y[left_lane_indices]
        rightx = x[right_lane_indices]
        righty = y[right_lane_indices]

        try:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
        except:

            left_fit = np.array([0, 0, 0])
            right_fit = np.array([0, 0, 0])

        return left_fit, right_fit

    def _find_points(self, img_shape, left_fit, right_fit):
        """Find points for the lane curves."""
        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

        return pts_left, pts_right

    def _fill_curves(self, img_shape, pts_left, pts_right):
        """Fill area between curves."""
        pts = np.hstack((pts_left, pts_right))
        img = np.zeros((img_shape[0], img_shape[1], 3), dtype="uint8")
        cv2.fillPoly(img, np.int32([pts]), (0, 0, 255))
        return img

    def _draw_curves(self, img, pts_left, pts_right):
        """Draw the curves of detected lanes on an image."""
        img_3ch = self._one_to_three_channel(img)
        cv2.polylines(
            img_3ch,
            np.int32([pts_left]),
            isClosed=False,
            color=(0, 0, 255),
            thickness=10,
        )
        cv2.polylines(
            img_3ch,
            np.int32([pts_right]),
            isClosed=False,
            color=(0, 255, 255),
            thickness=10,
        )
        return img_3ch

    def _one_to_three_channel(self, binary):
        """Convert a one channel image into a three channel image."""
        if len(binary.shape) == 3:
            return binary

        img = np.zeros((binary.shape[0], binary.shape[1], 3), dtype="uint8")
        img[:, :, 0] = binary
        img[:, :, 1] = binary
        img[:, :, 2] = binary
        return img

    def _radius_of_curvature(self, img, left_fit, right_fit):
        """Calculate radius of curvature."""
        y_eval = img.shape[0] / 2

        try:
            left_radius = (
                (1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5
            ) / abs(2 * left_fit[0])
            right_radius = (
                (1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5
            ) / abs(2 * right_fit[0])

            if left_fit[0] > 0:
                left_radius = -left_radius
            if right_fit[0] > 0:
                right_radius = -right_radius

            avg_radius = (left_radius + right_radius) / 2
        except:
            left_radius = 0
            right_radius = 0
            avg_radius = 0

        return round(left_radius, 2), round(right_radius, 2), round(avg_radius, 2)

    def _information_window(self, left_radius, right_radius, avg_radius):
        """Create information window with radius values."""

        window = np.zeros((170, 1280, 3), dtype="uint8")
        window[:, :, 0] = 249
        window[:, :, 1] = 242
        window[:, :, 2] = 227

        text1 = "(1) : Detected white and yellow markings, (2) : Warped image, (3) : Curve fitting"
        window = cv2.putText(
            window,
            text1,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        text2 = f"Left Curvature : {left_radius}, Right Curvature : {right_radius}"
        window = cv2.putText(
            window,
            text2,
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        text3 = f"Average Curvature : {avg_radius}"
        window = cv2.putText(
            window,
            text3,
            (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        text4 = "ANPR System with Lane Detection"
        window = cv2.putText(
            window,
            text4,
            (800, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        return window

    def _set_offset(self, img, offset):
        """Add extra blank space in front of the image."""
        blank = np.zeros((img.shape[0], offset, 3), dtype="uint8")
        img = np.concatenate((blank, img), axis=1)
        return img

    def _concatenate(self, img1, img2, img3, img4, img5):
        """Concatenate various images to one image.

        This follows the approach from curved_lane_detection.py example.
        """
        offset = 50
        img3 = self._set_offset(img3, offset)
        img4 = self._set_offset(img4, offset)

        img1 = cv2.resize(img1, (950, 550), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, (330, 180), interpolation=cv2.INTER_AREA)
        img3 = cv2.resize(img3, (165, 370), interpolation=cv2.INTER_AREA)
        img4 = cv2.resize(img4, (165, 370), interpolation=cv2.INTER_AREA)

        try:
            result = np.concatenate((img3, img4), axis=1)
            result = np.concatenate((img2, result))
            result = np.concatenate((img1, result), axis=1)
            result = np.concatenate((result, img5), axis=0)
        except ValueError as e:
            print(f"Error concatenating images: {e}")
            result = img1

        return result

    def _add_turn_info(self, img, radius):
        """Add turn information and section labels to the image."""

        result = img.copy()

        if abs(radius) >= 10000 or radius == 0:
            text = "Go Straight"
            color = (0, 255, 0)

        elif radius > 0:
            text = "Turn Right"
            color = (0, 0, 255)

        else:
            text = "Turn Left"
            color = (255, 0, 0)

        overlay = result.copy()
        overlay_area = overlay[20:80, 20:250]
        overlay_area[:] = (50, 50, 50)
        result[20:80, 20:250] = cv2.addWeighted(
            overlay_area, 0.7, result[20:80, 20:250], 0.3, 0
        )

        result = cv2.putText(
            result, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA
        )

        result = cv2.putText(
            result,
            "(1)",
            (1000, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        result = cv2.putText(
            result,
            "(2)",
            (1000, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        result = cv2.putText(
            result,
            "(3)",
            (1165, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        return result

    def _debug_show(self, title, image):
        """Show debug image if debug mode is enabled."""
        if self.debug_mode:
            if len(image.shape) == 2:

                display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                display_image = image.copy()

            cv2.imshow(title, display_image)
            print(f"Showing: {title} - Press any key to continue...")
            cv2.waitKey(1)
