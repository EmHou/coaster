import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy  # Keep original import name
import numpy as np
import pandas as pd
from typing_extensions import Literal


# Position of the points on the image that should be brought to (255, 255, 255) to restore white balance
DOT_POSITIONS = [
    (80, 30),  # Top-left LED
    (272, 33),  # Top-right LED
    (80, 210),  # Bottom-left LED
    (263, 220),  # Bottom-right LED
]

# ----- IMAGE PROCESSING CONSTANTS -----
# TODO: if camera has moved, manually set
# Define constants for center coordinates and height (optional overrides)
# If set to a value other than None, these will be used for all images.
# Otherwise, the center and height of each specific image will be used.
TARGET_CENTER_X = 182
TARGET_CENTER_Y = 122
TARGET_HEIGHT = 150

# Used for white balance calculation. Do not touch unless the CLI prompts you to do so. Doesn't seem to matter too much
DOT_RADIUS = 8

# 0.7 at home. If the room is brighter, go higher
# TODO: perhaps manually set
CEILING_MULTIPLIER = 0.7

# ----- DRINK CLASSIFICATION CONSTANTS -----
# TODO: manually set
# if the average RGB is above this threshold, it's definitely coffee
COFFEE_AVG_RGB_THRESHOLD = (110, 110, 110)

# if the red ratio is above this threshold and the above is not true, it's definitely fruit_punch
# red ratio is R / avg(G, B), i.e. how much more red there is than the average of green and blue
FRUIT_PUNCH_R_RATIO_THRESHOLD = 1.7

# if the RMSD is above this threshold and the above are not true, it's definitely water
# RMSD is bigger for water than for empty because of the LED reflection on the surface (the "halo" around the center)
WATER_RMSD_THRESHOLD = 18

# if the above are not true, it's empty

# All possible drinks
Drink = Literal["coffee", "water", "fruit_punch", "empty", "nothing"]


# DO NOT SET MANUALLY!
ceiling_r, ceiling_g, ceiling_b = None, None, None


def set_ceiling_color(empty_image_path: str):
    """
    Reads the empty image and sets the global ceiling color variables
    based on the pixel color at (TARGET_CENTER_X, TARGET_CENTER_Y).
    Make sure you're passing an ACTUALLY EMPTY image without any liquid in the receptacle!
    """
    global ceiling_r, ceiling_g, ceiling_b

    # Convert Path object to string for cv2.imread
    print(f"Setting ceiling color from: {empty_image_path}")

    img = cv2.imread(empty_image_path)

    if img is None:
        raise Exception(f"Error: Could not read image at {empty_image_path}")

    h, w = img.shape[:2]

    # Check if target coordinates are defined
    if TARGET_CENTER_X is None or TARGET_CENTER_Y is None:
        raise ValueError(
            "Error: TARGET_CENTER_X and TARGET_CENTER_Y must be defined to set the ceiling color."
        )

    if (
        TARGET_CENTER_X < 0
        or TARGET_CENTER_X >= w
        or TARGET_CENTER_Y < 0
        or TARGET_CENTER_Y >= h
    ):
        raise ValueError(
            f"Error: Target center ({TARGET_CENTER_X}, {TARGET_CENTER_Y}) is outside the bounds "
            f"of the image ({w}x{h}). Ceiling color not set."
        )

    # Get the BGR color at the target pixel (OpenCV uses y, x indexing)
    # Ensure coordinates are integers
    center_y = int(TARGET_CENTER_Y)
    center_x = int(TARGET_CENTER_X)
    center_pixel_bgr = img[center_y, center_x]
    b, g, r = center_pixel_bgr[0], center_pixel_bgr[1], center_pixel_bgr[2]

    # Assign to global variables
    ceiling_r, ceiling_g, ceiling_b = r, g, b

    print(
        f"Ceiling color successfully set to: R={ceiling_r}, G={ceiling_g}, B={ceiling_b}"
    )


def _white_balance_image(img_path: str, output_path: str) -> np.ndarray:
    """
    White balances an image based on the ceiling color and saves it.

    Args:
        img_path: Path to the input image.
        white_balanced_path: Path where the white-balanced image will be saved.

    Returns:
        The white-balanced image as a NumPy array.
    """

    if ceiling_r is None or ceiling_g is None or ceiling_b is None:
        raise Exception(
            "Error: Ceiling color not set. Please call the set_ceiling_color(empty_image_path) function first."
        )

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Check if the path includes a directory
        os.makedirs(output_dir, exist_ok=True)

    print(
        f"Starting white balance processing for {img_path} based on average dot color..."
    )

    # Read the input image
    img = cv2.imread(img_path)

    # Check if image loading was successful
    if img is None:
        raise Exception(f"Error: Could not read image at {img_path}")

    h, w = img.shape[:2]
    dot_avg_colors_bgr = []
    valid_dots = 0
    sum_avg_b, sum_avg_g, sum_avg_r = 0.0, 0.0, 0.0

    # Calculate average color under each specified dot
    for i, position in enumerate(DOT_POSITIONS):
        # Ensure the circle centered at position with DOT_RADIUS is fully within image bounds
        if not (
            DOT_RADIUS <= position[0] < w - DOT_RADIUS
            and DOT_RADIUS <= position[1] < h - DOT_RADIUS
        ):
            print(
                f"Warning: Dot {i+1} at {position} is too close to the edge or outside bounds for image {img_path} ({w}x{h}). Skipping this dot. To prevent this, set DOT_RADIUS to a smaller value."
            )
            continue

        # Create a mask for the current dot
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, position, DOT_RADIUS, 255, -1)  # Use DOT_RADIUS

        # Calculate the mean color (BGR) within the mask
        # Check if mask has any non-zero pixels to avoid division by zero in cv2.mean if mask is empty
        if cv2.countNonZero(mask) > 0:
            mean_bgr = cv2.mean(img, mask=mask)[
                :3
            ]  # Returns (B, G, R, Alpha) if 4 channels, slice to get BGR

            # Check if mean calculation was successful (mean_bgr might contain NaNs if mask is outside image)
            if not np.isnan(mean_bgr).any():
                dot_avg_colors_bgr.append(mean_bgr)
                sum_avg_b += mean_bgr[0]
                sum_avg_g += mean_bgr[1]
                sum_avg_r += mean_bgr[2]
                valid_dots += 1
            else:
                print(
                    f"Warning: Mean calculation failed for dot {i+1} at {position} in { img_path} (likely mask issue). Skipping this dot."
                )
        else:
            # This case should ideally not happen with the boundary check above, but added for robustness
            print(
                f"Warning: Mask for dot {i+1} at {position} in { img_path} resulted in zero pixels. Skipping this dot."
            )

    # Check if we have enough valid dots to calculate an overall average
    if valid_dots < len(DOT_POSITIONS):
        print(
            f"Warning: Only {valid_dots}/{len(DOT_POSITIONS)} dots were valid for calculating average color in { img_path}."
        )
        # Proceed if at least one dot is valid.
        if valid_dots == 0:
            # Raise an exception instead of just printing, as processing cannot continue meaningfully.
            raise ValueError(
                f"Error: No valid dots found for {img_path}. Cannot calculate average color. Cannot perform white balance."
            )

    # Calculate the overall average BGR across the valid dots
    # Avoid division by zero if valid_dots is somehow zero despite the check above
    overall_avg_b = (sum_avg_b / valid_dots) if valid_dots > 0 else 0
    overall_avg_g = (sum_avg_g / valid_dots) if valid_dots > 0 else 0
    overall_avg_r = (sum_avg_r / valid_dots) if valid_dots > 0 else 0
    print(
        f"  {img_path}: Avg dot color (BGR): ({overall_avg_b:.1f}, {overall_avg_g:.1f}, {overall_avg_r:.1f}) based on {valid_dots} dots."
    )

    # Calculate scaling factors to make the average dot color white (255, 255, 255)
    # Add a small epsilon to prevent division by zero if an average channel value is 0
    epsilon = 1e-6
    scale_b = 255.0 / (overall_avg_b + epsilon)
    scale_g = 255.0 / (overall_avg_g + epsilon)
    scale_r = 255.0 / (overall_avg_r + epsilon)

    # Apply scaling and subtraction to the entire image
    # Convert image to float32 for calculations to prevent precision loss and allow values > 255 temporarily
    img_float = img.astype(np.float32)

    # Split channels
    b, g, r = cv2.split(img_float)

    # 1. Scale channels for white balance
    b_scaled = b * scale_b
    g_scaled = g * scale_g
    r_scaled = r * scale_r

    # 2. Subtract the previously set ceiling color (BGR order)
    b_final = b_scaled - ceiling_b * CEILING_MULTIPLIER
    g_final = g_scaled - ceiling_g * CEILING_MULTIPLIER
    r_final = r_scaled - ceiling_r * CEILING_MULTIPLIER

    # 3. Clip values to the valid range [0, 255]
    # np.clip is efficient for this
    b_final = np.clip(b_final, 0, 255)
    g_final = np.clip(g_final, 0, 255)
    r_final = np.clip(r_final, 0, 255)

    # Merge channels back
    img_white_balanced_float = cv2.merge((b_final, g_final, r_final))

    # Convert back to uint8 for display and saving
    img_white_balanced = img_white_balanced_float.astype(np.uint8)

    # --- Save the white balanced image before adding overlays ---
    try:
        cv2.imwrite(output_path, img_white_balanced)
        print(f"  Successfully saved white-balanced image to {output_path}")
    except Exception as e:
        # Raise the exception to make the failure explicit
        raise IOError(f"Error saving image {output_path}: {e}") from e
    # --- End of saving ---

    return img_white_balanced


# Define a class to hold the results
@dataclass
class ImageAndValues:
    """Holds the processed image and calculated statistical values."""

    image: Optional[
        numpy.ndarray
    ]  # The cropped image (numpy array) or None if crop failed
    avg_rgb: Optional[Tuple[float, float, float]]  # (R, G, B)
    std_rgb: Optional[Tuple[float, float, float]]  # (R, G, B)
    r_ratio: Optional[
        float
    ]  # Ratio of red to other colors - used in fruit_punch determination, i.e. R / (avg(G, B))
    diff_rgb: Optional[
        Tuple[float, float, float]
    ]  # Absolute diff from center (R, G, B) or None
    rmsd_rgb: Optional[Tuple[float, float, float]]  # RMSD from center (R, G, B) or None


from pathlib import Path  # Ensure Path is available


def _crop_and_calculate_values(
    img_path: str, output_path: str
) -> Optional[ImageAndValues]:
    """
    Reads an image, white-balances it, rotates, crops, saves the cropped version,
    calculates statistics, and returns them.

    Args:
        img_path: Path to the original input image.
        output_path: Path where the processed (cropped) image should be saved.

    Returns:
        An ImageAndValues object containing the cropped image and calculated stats,
        or None if the image cannot be read or white balancing fails.
        Returns ImageAndValues with None fields if cropping or calculations fail
        at specific steps after white balancing.
    """
    # Assume these constants are defined in the global scope where this function is called
    # Use globals().get() for safer access, providing defaults if they don't exist or are None
    TARGET_CENTER_X = globals().get("TARGET_CENTER_X")
    TARGET_CENTER_Y = globals().get("TARGET_CENTER_Y")
    TARGET_HEIGHT = globals().get("TARGET_HEIGHT")

    original_filename = os.path.basename(img_path)

    # --- White Balance Step ---
    try:
        p = Path(img_path)
        # Construct path for the white-balanced image
        # Assumes the input path structure allows navigating up one level
        # e.g., if img_path is 'input_data/run1/image.jpg', wb_dir becomes 'input_data/white_balanced'
        wb_dir = p.parent.parent / "white_balanced"
        # Ensure the target directory exists (though _white_balance_image might also do this)
        wb_dir.mkdir(parents=True, exist_ok=True)
        white_balanced_output_path = str(wb_dir / p.name)

        print(
            f"  White balancing {original_filename} to {white_balanced_output_path}..."
        )
        # Call the white balance function - it saves the image to white_balanced_output_path
        _white_balance_image(img_path, white_balanced_output_path)

        # Update img_path to point to the white-balanced image for subsequent steps
        img_path = white_balanced_output_path
        filename = os.path.basename(img_path)  # Update filename if needed for logging
        print(f"  Successfully created white-balanced image: {filename}")

    except Exception as e:
        print(f"Error during white balancing for {original_filename}: {e}")
        # If white balancing fails, we cannot proceed.
        return None
    # --- End White Balance Step ---

    # Read the WHITE-BALANCED image
    img = cv2.imread(img_path)

    # Check if image loading was successful (could fail if wb image wasn't saved correctly)
    if img is None:
        print(
            f"Warning: Could not read white-balanced image {filename} at {img_path}. Skipping."
        )
        return None

    # Get image dimensions (of the white-balanced image)
    h_img, w_img = img.shape[:2]

    # Determine the center coordinates and height to use
    # Use constants if provided, otherwise use image-specific dimensions
    center_x_defined = (
        int(TARGET_CENTER_X) if TARGET_CENTER_X is not None else w_img // 2
    )
    center_y_defined = (
        int(TARGET_CENTER_Y) if TARGET_CENTER_Y is not None else h_img // 2
    )
    h_for_crop = int(TARGET_HEIGHT) if TARGET_HEIGHT is not None else h_img

    # --- Get Center Pixel Color (from WHITE-BALANCED image) ---
    # Note: This color is now from the white-balanced image, not the original.
    # If the original center pixel color is needed, it should be read before white balancing.
    center_pixel_bgr = None
    center_coords_valid = False
    if 0 <= center_x_defined < w_img and 0 <= center_y_defined < h_img:
        center_pixel_bgr = img[center_y_defined, center_x_defined]
        center_coords_valid = True
        print(
            f"  Center pixel ({center_x_defined},{center_y_defined}) BGR from WB image: {center_pixel_bgr} for {filename}"
        )
    else:
        print(
            f"Warning: Target center ({center_x_defined}, {center_y_defined}) is outside WB image bounds ({w_img}x{h_img}) for {filename}. Cannot calculate diff/RMSD from center."
        )

    # --- Rotation (of the white-balanced image) ---
    M = cv2.getRotationMatrix2D((center_x_defined, center_y_defined), -45, 1.0)
    rotated_img = cv2.warpAffine(img, M, (w_img, h_img))

    # --- Cropping (from the rotated white-balanced image) ---
    side_length = int(h_for_crop / numpy.sqrt(2))
    crop_x1_raw = int(center_x_defined - side_length / 2)
    crop_y1_raw = int(center_y_defined - side_length / 2)
    crop_x2_raw = crop_x1_raw + side_length
    crop_y2_raw = crop_y1_raw + side_length

    x1 = max(0, crop_x1_raw)
    y1 = max(0, crop_y1_raw)
    x2 = min(w_img, crop_x2_raw)
    y2 = min(h_img, crop_y2_raw)

    cropped_img = None
    is_valid_crop = False
    if x1 >= x2 or y1 >= y2:
        print(
            f"Warning: Calculated crop dimensions [{x1}:{x2}, {y1}:{y2}] are invalid for {filename}. Skipping crop and save."
        )
        is_valid_crop = False
    else:
        cropped_img = rotated_img[y1:y2, x1:x2]
        # Double check if crop actually yielded an image
        if cropped_img is not None and cropped_img.size > 0:
            is_valid_crop = True
        else:
            print(
                f"Warning: Cropping resulted in empty image for {filename}. Skipping save."
            )
            is_valid_crop = False
            cropped_img = None  # Ensure it's None

    # Initialize results
    avg_rgb_res = None
    std_rgb_res = None
    r_ratio_res = None
    diff_rgb_res = None
    rmsd_rgb_res = None

    # --- Calculate Stats and Save Cropped Image ---
    # The output_path here is for the FINAL CROPPED image, not the intermediate WB one.
    if is_valid_crop and cropped_img is not None:
        try:
            # Ensure output directory exists before saving the cropped image
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"  Created output directory for cropped image: {output_dir}")

            cv2.imwrite(output_path, cropped_img)
            print(f"  Successfully saved final cropped image to {output_path}")

            # Calculate mean and standard deviation (OpenCV returns BGR order)
            mean_val, stddev_val = cv2.meanStdDev(cropped_img)
            avg_b, avg_g, avg_r = mean_val.flatten()
            std_b, std_g, std_r = stddev_val.flatten()
            avg_rgb_res = (avg_r, avg_g, avg_b)  # Store as RGB
            std_rgb_res = (std_r, std_g, std_b)  # Store as RGB

            # --- Calculate Red Ratio ---
            avg_gb = (avg_g + avg_b) / 2.0
            if avg_gb > 1e-6:
                r_ratio_res = avg_r / avg_gb
            else:
                r_ratio_res = None  # Indicate N/A
                print(
                    f"  Warning: Average G+B is near zero for {filename}. Cannot calculate Red Ratio."
                )

            # --- Calculate Difference and RMSD from Center ---
            # Uses center_pixel_bgr obtained from the white-balanced image earlier
            if center_coords_valid and center_pixel_bgr is not None:
                center_bgr_np = numpy.array(center_pixel_bgr, dtype=numpy.float64)
                mean_bgr_np = mean_val.flatten()  # BGR order

                # Difference
                diff_bgr = numpy.abs(mean_bgr_np - center_bgr_np)
                diff_b, diff_g, diff_r = diff_bgr[0], diff_bgr[1], diff_bgr[2]
                diff_rgb_res = (diff_r, diff_g, diff_b)  # Store as RGB

                # RMSD
                cropped_float = cropped_img.astype(numpy.float64)
                pixel_diffs = cropped_float - center_bgr_np
                squared_pixel_diffs = numpy.square(pixel_diffs)
                mean_sq_diff_per_channel = numpy.mean(squared_pixel_diffs, axis=(0, 1))
                rmsd_per_channel = numpy.sqrt(mean_sq_diff_per_channel)  # BGR order
                rmsd_b, rmsd_g, rmsd_r = (
                    rmsd_per_channel[0],
                    rmsd_per_channel[1],
                    rmsd_per_channel[2],
                )
                rmsd_rgb_res = (rmsd_r, rmsd_g, rmsd_b)  # Store as RGB
            else:
                # Center was OOB or pixel couldn't be read, diff/RMSD are None
                diff_rgb_res = None
                rmsd_rgb_res = None
                print(
                    f"  Skipping Diff/RMSD calculation for {filename} due to invalid center in WB image."
                )

        except Exception as e:
            print(f"Error processing stats or saving cropped {filename}: {e}")
            # Reset values if error occurred during calculation/saving
            cropped_img = None  # Indicate failure by setting image to None
            avg_rgb_res = None
            std_rgb_res = None
            r_ratio_res = None
            diff_rgb_res = None
            rmsd_rgb_res = None

    # Return the results object
    return ImageAndValues(
        image=cropped_img,  # Will be None if crop invalid or error occurred
        avg_rgb=avg_rgb_res,
        std_rgb=std_rgb_res,
        r_ratio=r_ratio_res,
        diff_rgb=diff_rgb_res,
        rmsd_rgb=rmsd_rgb_res,
    )


def analyze_image(img_path: str, output_path: str) -> Optional[Drink]:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image at {img_path}")
        return None

    for idx, (x, y) in enumerate(DOT_POSITIONS):
        b, g, r = img[y, x]
        if r < 100 or g < 100 or b < 100:
            print(f"Receptacle check failed at DOT_POSITION {idx} ({x}, {y}): RGB=({r}, {g}, {b}). Assuming there's no receptacle on the coaster.")
            return "nothing"

    values = _crop_and_calculate_values(img_path, output_path)
    if values is None:
        print(
            "  Analysis skipped: Failed to process image or calculate initial values."
        )
        return None
    if values.avg_rgb is None:
        print("  Analysis skipped: Average RGB could not be calculated.")
        return None  # Cannot classify without average color

    # Unpack values for easier access, checking for None where necessary
    avg_r, avg_g, avg_b = values.avg_rgb
    r_ratio = values.r_ratio
    rmsd_rgb = values.rmsd_rgb

    # 1. Check for Coffee
    if (
        avg_r > COFFEE_AVG_RGB_THRESHOLD[0]
        and avg_g > COFFEE_AVG_RGB_THRESHOLD[1]
        and avg_b > COFFEE_AVG_RGB_THRESHOLD[2]
    ):
        print(
            f"  Classified as: coffee (Avg RGB {avg_r:.1f},{avg_g:.1f},{avg_b:.1f} > Threshold {COFFEE_AVG_RGB_THRESHOLD})"
        )
        return "coffee"

    # 2. Check for Juice (only if not coffee)
    if r_ratio is not None and r_ratio > FRUIT_PUNCH_R_RATIO_THRESHOLD:
        print(
            f"  Classified as: fruit_punch (R Ratio {r_ratio:.2f} > Threshold {FRUIT_PUNCH_R_RATIO_THRESHOLD})"
        )
        return "fruit_punch"

    # 3. Check for Water (only if not coffee or fruit_punch)
    if rmsd_rgb is not None:
        rmsd_r, rmsd_g, rmsd_b = rmsd_rgb
        # Calculate average RMSD across channels
        avg_rmsd = (rmsd_r + rmsd_g + rmsd_b) / 3.0
        if avg_rmsd > WATER_RMSD_THRESHOLD:
            print(
                f"  Classified as: water (Avg RMSD {avg_rmsd:.2f} > Threshold {WATER_RMSD_THRESHOLD})"
            )
            return "water"
        else:
            # This path leads to 'empty' if RMSD is calculated but below threshold
            print(
                f"  Classification: Not water (Avg RMSD {avg_rmsd:.2f} <= Threshold {WATER_RMSD_THRESHOLD})"
            )
    else:
        # This path leads to 'empty' if RMSD could not be calculated
        print(f"  Classification: RMSD not available, cannot check for water.")

    # 4. Default to Empty
    print("  Classified as: empty (No other conditions met)")
    return "empty"


if __name__ == "__main__":
    set_ceiling_color("data/empty.jpg")
    # values = _crop_and_calculate_values("data/fruit_punch.jpg", "final/fruit_punch.jpg")
    # if values:
    #     print("\nCalculated Values (excluding image):")

    #     # Helper to format tuples of floats or return 'N/A' if None
    #     def format_tuple(t):
    #         if t is None:
    #             return "N/A"
    #         # Assuming RGB tuples
    #         if len(t) == 3 and all(isinstance(x, (float, int, np.number)) for x in t):
    #             return f"({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})"
    #         return str(t)  # Fallback for other tuple types

    #     print(f"  Average RGB: {format_tuple(values.avg_rgb)}")
    #     print(f"  Std Dev RGB: {format_tuple(values.std_rgb)}")
    #     print(
    #         f"  R Ratio: {values.r_ratio:.2f}" if values.r_ratio is not None else "N/A"
    #     )
    #     print(f"  Diff RGB: {format_tuple(values.diff_rgb)}")
    #     print(f"  RMSD RGB: {format_tuple(values.rmsd_rgb)}")
    # else:
    #     print("\nFailed to calculate values.")

    drink_type = analyze_image("data/nothing.jpg", "final/output.jpg")
    print(f"FINAL DRINK TYPE: {drink_type}")
