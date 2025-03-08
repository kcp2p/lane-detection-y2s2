"""
License plate detector using traditional computer vision techniques.
"""

import cv2
import numpy as np


class LicensePlateDetector:
    """License plate detector using traditional computer vision techniques."""

    def __init__(self, debug_mode=False):
        """Initialize the detector.

        Args:
            debug_mode (bool): If True, shows intermediate processing steps
        """
        self.debug_mode = debug_mode

    def order_points(self, pts):
        """Order points in a quadrilateral (top-left, top-right, bottom-right, bottom-left)."""
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def four_point_transform(self, image, pts):
        """Apply perspective transform to obtain a top-down view of a plate."""
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array(
            [
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1],
            ],
            dtype="float32",
        )

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped

    def automatic_brightness_contrast(self, image, clip_hist_percent=10):
        """Automatically adjust brightness and contrast of an image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_size = len(hist)

        accumulator = []
        accumulator.append(float(hist[0].item()))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index - 1] + float(hist[index].item()))

        maximum = accumulator[-1]
        clip_hist_percent *= maximum / 100.0
        clip_hist_percent /= 2.0

        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        maximum_gray = hist_size - 1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return auto_result, alpha, beta

    def detect(self, img_rgb):
        """Detect license plates in an image.

        Args:
            img_rgb: Input BGR image

        Returns:
            tuple: (Annotated image, List of detected license plate images)
        """
        img = img_rgb.copy()
        input_height = img_rgb.shape[0]
        input_width = img_rgb.shape[1]

        hsv_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

        low_yellow = np.array([20, 100, 100])
        high_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv_frame, low_yellow, high_yellow)
        yellow = cv2.bitwise_and(yellow_mask, yellow_mask, mask=yellow_mask)

        self._debug_show("1. Yellow Color Detection", yellow)

        k = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, k)

        self._debug_show("2. Closing Morphology", closing)

        contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        crops = []

        imgray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if (
                h * 6 > w > 2 * h
                and h > 0.1 * w
                and w * h > input_height * input_width * 0.0001
            ):

                try:
                    crop_img = img_rgb[y : y + h, x - round(w / 10) : x]
                    crop_img = crop_img.astype("uint8")

                    hsv_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
                    low_blue = np.array([100, 150, 0])
                    high_blue = np.array([140, 255, 255])
                    blue_mask = cv2.inRange(hsv_crop, low_blue, high_blue)
                    blue_summation = blue_mask.sum()
                except:
                    blue_summation = 0

                if blue_summation > 550:

                    crop_img_yellow = img_rgb[y : y + h, x : x + w]
                    crop_img_yellow = crop_img_yellow.astype("uint8")

                    hsv_crop_yellow = cv2.cvtColor(crop_img_yellow, cv2.COLOR_BGR2HSV)
                    yellow_mask = cv2.inRange(hsv_crop_yellow, low_yellow, high_yellow)
                    yellow_summation = yellow_mask.sum()

                    if (
                        yellow_summation
                        > 255 * crop_img.shape[0] * crop_img.shape[0] * 0.4
                    ):

                        crop_gray = imgray[y : y + h, x : x + w]
                        crop_gray = crop_gray.astype("uint8")

                        th = cv2.adaptiveThreshold(
                            crop_gray,
                            255,
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY,
                            11,
                            2,
                        )
                        char_contours, _ = cv2.findContours(
                            th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                        )

                        chars = 0
                        for c in char_contours:
                            area2 = cv2.contourArea(c)
                            x2, y2, w2, h2 = cv2.boundingRect(c)
                            if (
                                w2 * h2 > h * w * 0.01
                                and h2 > w2
                                and area2 < h * w * 0.9
                            ):
                                chars += 1

                        if 20 > chars > 4:

                            rect = cv2.minAreaRect(cnt)
                            box = cv2.boxPoints(rect)
                            box = np.array(box, dtype=np.intp)

                            pts = np.array(box)
                            warped = self.four_point_transform(img, pts)
                            crops.append(warped)

                            cv2.putText(
                                img_rgb,
                                "LP",
                                (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255),
                                2,
                                cv2.LINE_AA,
                            )
                            cv2.drawContours(img_rgb, [box], 0, (0, 0, 255), 2)

        return img_rgb, crops

    def process_plate(self, src):
        """Process a detected license plate for OCR.

        Args:
            src: Source image of license plate

        Returns:
            Processed binary image ready for OCR
        """
        self._debug_show("3. Detected Plate", src)

        adjusted, _, _ = self.automatic_brightness_contrast(src)
        self._debug_show("4. Brightness & Contrast Adjustment", adjusted)

        gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
        self._debug_show("5. Grayscale", gray)

        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self._debug_show("6. Thresholded", th)

        return th

    def _debug_show(self, title, image):
        """Show debug image if debug mode is enabled."""
        if self.debug_mode:
            cv2.imshow(title, image)
            print(f"Showing: {title} - Press any key to continue...")
            cv2.waitKey(0)
