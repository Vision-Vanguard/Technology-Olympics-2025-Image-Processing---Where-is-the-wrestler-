import cv2
import numpy as np


def advanced_color_analysis(img):
    """Advanced multi-color space analysis for precise wrestler detection."""
    # Convert to multiple color spaces for comprehensive analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # Enhanced skin detection with multiple approaches
    # HSV skin detection with adaptive ranges
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # Calculate adaptive skin ranges based on image statistics
    h_mean, h_std = np.mean(h_channel), np.std(h_channel)
    s_mean, s_std = np.mean(s_channel), np.std(s_channel)

    # Multiple HSV skin ranges for different lighting conditions
    skin_ranges_hsv = [
        ([0, max(20, int(s_mean - s_std)), max(50, int(v_channel.mean() - v_channel.std()))],
         [25, 255, 255]),
        ([0, 30, 80], [20, 180, 255]),
        ([0, 10, 60], [25, 150, 255])
    ]

    skin_masks_hsv = []
    for lower, upper in skin_ranges_hsv:
        mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        skin_masks_hsv.append(mask)

    # LAB color space skin detection (more robust for varying lighting)
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]

    # Multiple LAB ranges for comprehensive skin detection
    skin_ranges_lab = [
        ([0, 130, 130], [255, 173, 127]),
        ([20, 128, 125], [255, 175, 135]),
        ([0, 125, 120], [255, 180, 140])
    ]

    skin_masks_lab = []
    for lower, upper in skin_ranges_lab:
        mask = cv2.inRange(lab, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        skin_masks_lab.append(mask)

    # YUV skin detection
    skin_mask_yuv = cv2.inRange(yuv, np.array([0, 77, 133], dtype=np.uint8),
                                np.array([255, 127, 173], dtype=np.uint8))

    # Combine all skin masks
    skin_mask = np.zeros_like(hsv[:, :, 0])
    for mask in skin_masks_hsv + skin_masks_lab + [skin_mask_yuv]:
        skin_mask = cv2.bitwise_or(skin_mask, mask)

    # Enhanced clothing detection with multiple ranges
    # Red clothing detection (multiple ranges for different lighting)
    red_ranges = [
        ([0, 50, 50], [10, 255, 255]),
        ([170, 50, 50], [180, 255, 255]),
        ([0, 30, 100], [15, 200, 255]),
        ([165, 30, 100], [180, 200, 255]),
        ([0, 70, 80], [8, 255, 255])
    ]

    red_mask = np.zeros_like(hsv[:, :, 0])
    for lower, upper in red_ranges:
        mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        red_mask = cv2.bitwise_or(red_mask, mask)

    # Blue clothing detection
    blue_ranges = [
        ([100, 50, 50], [130, 255, 255]),
        ([90, 30, 80], [140, 200, 255]),
        ([110, 80, 30], [125, 255, 200]),
        ([95, 40, 60], [135, 220, 255])
    ]

    blue_mask = np.zeros_like(hsv[:, :, 0])
    for lower, upper in blue_ranges:
        mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        blue_mask = cv2.bitwise_or(blue_mask, mask)

    # Additional colors (green, yellow, purple, etc.)
    other_ranges = [
        ([40, 40, 40], [80, 255, 255]),  # Green
        ([20, 100, 100], [30, 255, 255]),  # Yellow
        ([140, 50, 50], [170, 255, 255]),  # Purple/Magenta
        ([80, 30, 80], [100, 200, 255])   # Cyan
    ]

    other_mask = np.zeros_like(hsv[:, :, 0])
    for lower, upper in other_ranges:
        mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        other_mask = cv2.bitwise_or(other_mask, mask)

    # Combine all color evidence
    combined_color_mask = cv2.bitwise_or(cv2.bitwise_or(skin_mask, red_mask), blue_mask)
    combined_color_mask = cv2.bitwise_or(combined_color_mask, other_mask)

    return combined_color_mask


def texture_analysis(img):
    """Advanced texture analysis for wrestler detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Local Binary Pattern (LBP) approximation
    # Calculate LBP-like features using local comparisons
    kernel_size = 3
    lbp_mask = np.zeros_like(gray)

    # Simple LBP calculation
    for i in range(1, gray.shape[0] - 1):
        for j in range(1, gray.shape[1] - 1):
            center = gray[i, j]
            neighbors = [
                gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                gray[i+1, j-1], gray[i, j-1]
            ]

            # Count how many neighbors are greater than center
            count = sum(1 for neighbor in neighbors if neighbor > center)
            lbp_mask[i, j] = min(255, count * 32)  # Scale to 0-255

    # Texture-based mask focusing on regions with human-like texture
    texture_mask = cv2.threshold(lbp_mask, 80, 255, cv2.THRESH_BINARY)[1]

    return texture_mask


def gradient_analysis(img):
    """Advanced gradient-based edge and boundary analysis."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Multi-scale gradient analysis
    # Sobel gradients in X and Y directions
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Gradient magnitude and direction
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_direction = np.arctan2(sobel_y, sobel_x)

    # Normalize gradient magnitude
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Laplacian for blob detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Combine gradient information
    gradient_mask = cv2.bitwise_or(gradient_magnitude, laplacian)
    gradient_mask = cv2.threshold(gradient_mask, 50, 255, cv2.THRESH_BINARY)[1]

    return gradient_mask


def superpixel_segmentation(img):
    """Fast superpixel-based segmentation using optimized grid approach."""
    h, w = img.shape[:2]

    # Use simpler approach for faster execution
    num_superpixels = 100  # Reduced for speed
    superpixel_size = int(np.sqrt(h * w / num_superpixels))

    # Create simple grid-based superpixels
    labels = np.zeros((h, w), dtype=np.int32)
    label_id = 0

    for y in range(0, h, superpixel_size):
        for x in range(0, w, superpixel_size):
            y_end = min(y + superpixel_size, h)
            x_end = min(x + superpixel_size, w)
            labels[y:y_end, x:x_end] = label_id
            label_id += 1

    # Quick refinement using color similarity
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Find wrestler-like superpixels based on color
    superpixel_mask = np.zeros((h, w), dtype=np.uint8)

    for sp_id in range(label_id):
        mask = (labels == sp_id)
        if np.sum(mask) > 0:
            # Get mean color of superpixel
            mean_color = np.mean(hsv[mask], axis=0)

            # Check if it matches wrestler color profile
            if ((20 <= mean_color[0] <= 30) or  # Orange/yellow hues
                (0 <= mean_color[0] <= 10) or   # Red hues
                (160 <= mean_color[0] <= 180)) and mean_color[1] > 50:  # Good saturation
                superpixel_mask[mask] = 255

    return superpixel_mask


def intelligent_grabcut_with_guidance(img, color_mask, texture_mask, gradient_mask):
    """Advanced GrabCut with multiple guidance modalities."""
    height, width = img.shape[:2]

    # Multiple rectangle strategies
    rectangles = [
        (1, 1, width - 2, height - 2),      # Ultra-tight
        (3, 3, width - 6, height - 6),      # Tight
        (8, 8, width - 16, height - 16),    # Medium
        (15, 15, width - 30, height - 30)   # Loose
    ]

    grabcut_masks = []

    for i, rect in enumerate(rectangles):
        # Initialize GrabCut mask
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # Set initial mask based on color, texture, and gradient evidence
        # Combine evidence for probable foreground
        evidence_mask = cv2.bitwise_or(color_mask, texture_mask)
        evidence_mask = cv2.bitwise_or(evidence_mask, gradient_mask)

        # Clean evidence mask
        kernel = np.ones((3, 3), np.uint8)
        evidence_mask = cv2.morphologyEx(evidence_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        evidence_mask = cv2.morphologyEx(evidence_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Set probable foreground based on evidence
        mask[evidence_mask > 0] = cv2.GC_PR_FGD

        # Set definite background (image borders)
        border_width = max(2, min(10, min(width, height) // 50))
        mask[:border_width, :] = cv2.GC_BGD
        mask[-border_width:, :] = cv2.GC_BGD
        mask[:, :border_width] = cv2.GC_BGD
        mask[:, -border_width:] = cv2.GC_BGD

        # Apply GrabCut with varying iterations
        iterations = [15, 12, 10, 8][i]
        try:
            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_MASK)
        except:
            # Fallback to rectangle-only initialization
            mask = np.zeros(img.shape[:2], np.uint8)
            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_RECT)

        # Convert to binary mask
        binary_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        grabcut_masks.append(binary_mask)

    return grabcut_masks


def advanced_post_processing(masks, img):
    """Advanced post-processing with ensemble and morphological operations."""
    height, width = img.shape[:2]

    # Weighted ensemble of GrabCut results
    weights = [0.4, 0.3, 0.2, 0.1]  # Favor tighter rectangles
    ensemble_mask = np.zeros((height, width), dtype=np.float32)

    for mask, weight in zip(masks, weights):
        ensemble_mask += mask.astype(np.float32) * weight

    # Convert to binary with optimized threshold
    binary_ensemble = (ensemble_mask > 0.3).astype(np.uint8) * 255

    # Advanced morphological operations
    # Use different kernel sizes based on image size
    base_size = max(2, min(width, height) // 200)

    # Closing to connect wrestler parts
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_size + 1, base_size + 1))
    closed_mask = cv2.morphologyEx(binary_ensemble, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Opening to remove noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_size, base_size))
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    return opened_mask


def intelligent_contour_analysis(mask, img):
    """Intelligent contour analysis with shape and size constraints."""
    height, width = img.shape[:2]
    image_area = height * width

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return mask

    final_mask = np.zeros_like(mask)

    # Analyze each contour
    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Dynamic area thresholds based on image size
        min_area = max(500, image_area * 0.001)  # At least 0.1% of image
        max_area = image_area * 0.7  # At most 70% of image

        if min_area <= area <= max_area:
            # Calculate shape properties
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0

            # Human-like aspect ratio constraints
            if 0.2 <= aspect_ratio <= 4.0:
                # Calculate extent (contour area / bounding rectangle area)
                rect_area = w * h
                extent = float(area) / rect_area if rect_area > 0 else 0

                # Calculate solidity (contour area / convex hull area)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0

                # Accept contours with reasonable shape properties
                if extent > 0.2 and solidity > 0.3:
                    valid_contours.append(contour)

    # Draw valid contours
    if valid_contours:
        for contour in valid_contours:
            cv2.fillPoly(final_mask, [contour], 255)
    else:
        # If no contours pass strict filtering, use more lenient criteria
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max(200, image_area * 0.0005):
                cv2.fillPoly(final_mask, [contour], 255)

    return final_mask


def detect_wrestler(image_path: str) -> np.ndarray:
    """High-precision wrestler detection targeting 0.8-0.9 IoU with optimized algorithms."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    height, width = img.shape[:2]

    # Step 1: Enhanced color segmentation with multiple approaches
    color_mask = advanced_color_analysis(img)

    # Step 2: Edge-enhanced segmentation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Bilateral filter to preserve edges while smoothing
    filtered = cv2.bilateralFilter(img, 9, 75, 75)

    # Multi-threshold approach for better segmentation
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)

    # Create multiple masks for different lighting conditions
    masks = []

    # Skin detection with multiple ranges
    skin_lower1 = np.array([0, 20, 70], dtype=np.uint8)
    skin_upper1 = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask1 = cv2.inRange(hsv, skin_lower1, skin_upper1)

    skin_lower2 = np.array([0, 10, 60], dtype=np.uint8)
    skin_upper2 = np.array([25, 150, 255], dtype=np.uint8)
    skin_mask2 = cv2.inRange(hsv, skin_lower2, skin_upper2)

    # Combine skin masks
    skin_combined = cv2.bitwise_or(skin_mask1, skin_mask2)
    masks.append(skin_combined)

    # Red clothing detection
    red_lower1 = np.array([0, 50, 50], dtype=np.uint8)
    red_upper1 = np.array([10, 255, 255], dtype=np.uint8)
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)

    red_lower2 = np.array([170, 50, 50], dtype=np.uint8)
    red_upper2 = np.array([180, 255, 255], dtype=np.uint8)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)

    red_combined = cv2.bitwise_or(red_mask1, red_mask2)
    masks.append(red_combined)

    # Blue clothing detection
    blue_lower = np.array([100, 50, 50], dtype=np.uint8)
    blue_upper = np.array([130, 255, 255], dtype=np.uint8)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    masks.append(blue_mask)

    # Combine all masks
    combined_mask = np.zeros_like(gray)
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Step 3: Morphological operations to clean the mask
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Remove noise
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)

    # Fill gaps
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_large, iterations=3)

    # Step 4: Multiple GrabCut iterations with different strategies
    best_result = None
    best_score = 0

    # Strategy 1: Use entire combined mask as guidance
    result1 = apply_grabcut_strategy(img, combined_mask, strategy="full_mask")
    score1 = evaluate_mask_quality(result1, combined_mask, height, width)

    if score1 > best_score:
        best_result = result1
        best_score = score1

    # Strategy 2: Use largest connected component
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        contour_mask = np.zeros_like(combined_mask)
        cv2.fillPoly(contour_mask, [largest_contour], 255)

        result2 = apply_grabcut_strategy(img, contour_mask, strategy="largest_component")
        score2 = evaluate_mask_quality(result2, combined_mask, height, width)

        if score2 > best_score:
            best_result = result2
            best_score = score2

    # Strategy 3: Conservative rectangle approach
    if np.sum(combined_mask) > 0:
        coords = np.where(combined_mask > 0)
        y_min, y_max = np.min(coords[0]), np.max(coords[0])
        x_min, x_max = np.min(coords[1]), np.max(coords[1])

        # Expand rectangle
        margin = max(20, min(width, height) // 20)
        rect = (max(0, x_min - margin), max(0, y_min - margin),
                min(width, x_max + margin) - max(0, x_min - margin),
                min(height, y_max + margin) - max(0, y_min - margin))

        result3 = apply_grabcut_strategy(img, combined_mask, strategy="rectangle", rect=rect)
        score3 = evaluate_mask_quality(result3, combined_mask, height, width)

        if score3 > best_score:
            best_result = result3
            best_score = score3

    # Use best result or fallback
    if best_result is not None and best_score > 0.1:
        final_result = best_result
    else:
        final_result = combined_mask

    # Step 5: Final refinement
    # Additional morphological cleanup
    final_result = cv2.morphologyEx(final_result, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
    final_result = cv2.morphologyEx(final_result, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # Smooth edges
    final_result = cv2.medianBlur(final_result, 5)

    # Convert to RGB format
    output_image = cv2.cvtColor(final_result, cv2.COLOR_GRAY2BGR)
    return output_image


def apply_grabcut_strategy(img, guidance_mask, strategy="full_mask", rect=None):
    """Apply GrabCut with different strategies."""
    height, width = img.shape[:2]

    # Initialize GrabCut
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    if strategy == "full_mask":
        # Use guidance mask for probable foreground
        mask[guidance_mask > 0] = cv2.GC_PR_FGD

        # Set borders as background
        border = max(5, min(width, height) // 50)
        mask[:border, :] = cv2.GC_BGD
        mask[-border:, :] = cv2.GC_BGD
        mask[:, :border] = cv2.GC_BGD
        mask[:, -border:] = cv2.GC_BGD

        # Use loose rectangle
        margin = max(20, min(width, height) // 20)
        rect = (margin, margin, width - 2*margin, height - 2*margin)

        try:
            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_MASK)
        except:
            return guidance_mask

    elif strategy == "largest_component":
        # Similar to full_mask but with largest component only
        mask[guidance_mask > 0] = cv2.GC_FGD  # Definite foreground

        border = max(5, min(width, height) // 50)
        mask[:border, :] = cv2.GC_BGD
        mask[-border:, :] = cv2.GC_BGD
        mask[:, :border] = cv2.GC_BGD
        mask[:, -border:] = cv2.GC_BGD

        margin = max(15, min(width, height) // 25)
        rect = (margin, margin, width - 2*margin, height - 2*margin)

        try:
            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 12, cv2.GC_INIT_WITH_MASK)
        except:
            return guidance_mask

    elif strategy == "rectangle" and rect is not None:
        # Rectangle-only initialization
        try:
            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 8, cv2.GC_INIT_WITH_RECT)
        except:
            return guidance_mask

    # Extract result
    result = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
    return result


def evaluate_mask_quality(mask, reference_mask, height, width):
    """Evaluate mask quality based on various metrics."""
    if mask is None or np.sum(mask) == 0:
        return 0.0

    score = 0.0
    total_area = height * width
    mask_area = np.sum(mask > 0)

    # Size reasonableness (wrestlers should be 2-60% of image)
    size_ratio = mask_area / total_area
    if 0.02 <= size_ratio <= 0.6:
        score += 0.4
    elif 0.01 <= size_ratio <= 0.8:
        score += 0.2

    # Overlap with reference mask
    if np.sum(reference_mask) > 0:
        overlap = np.sum(cv2.bitwise_and(mask, reference_mask))
        overlap_ratio = overlap / np.sum(reference_mask)
        score += overlap_ratio * 0.6

    return score


def calculate_iou(predicted_mask, ground_truth_mask):
    """
    Calculate Intersection over Union (IoU) score between predicted and ground truth masks.
    """
    # Convert to binary if needed
    if len(predicted_mask.shape) == 3:
        pred_binary = cv2.cvtColor(predicted_mask, cv2.COLOR_RGB2GRAY)
    else:
        pred_binary = predicted_mask.copy()

    if len(ground_truth_mask.shape) == 3:
        gt_binary = cv2.cvtColor(ground_truth_mask, cv2.COLOR_RGB2GRAY)
    else:
        gt_binary = ground_truth_mask.copy()

    # Ensure binary masks
    pred_binary = (pred_binary > 128).astype(np.uint8) * 255
    gt_binary = (gt_binary > 128).astype(np.uint8) * 255

    # Calculate intersection and union
    intersection = cv2.bitwise_and(pred_binary, gt_binary)
    union = cv2.bitwise_or(pred_binary, gt_binary)

    # Count non-zero pixels
    intersection_area = cv2.countNonZero(intersection)
    union_area = cv2.countNonZero(union)

    # Calculate IoU
    if union_area == 0:
        return 1.0 if intersection_area == 0 else 0.0

    iou = intersection_area / union_area
    return iou


def evaluate_wrestler_detection(image_path, ground_truth_path):
    """
    Evaluate wrestler detection performance.
    """
    predicted_mask = detect_wrestler(image_path)
    ground_truth = cv2.imread(ground_truth_path)

    if ground_truth is None:
        raise ValueError(f"Could not load ground truth mask from {ground_truth_path}")

    iou_score = calculate_iou(predicted_mask, ground_truth)
    return predicted_mask, iou_score


def main():
    """
    Main function to test the advanced wrestler segmentation system.
    """
    test_images = ['1', '2', '3']
    image_dir = "image"
    mask_dir = "mask"

    print("🏆 REVOLUTIONARY Wrestler Segmentation System v2.0")
    print("Target: 0.8-0.9 IoU | Advanced CV Techniques")
    print("=" * 60)

    total_iou = 0.0
    image_count = len(test_images)

    for image_id in test_images:
        image_path = f"{image_dir}/{image_id}.jpg"
        mask_path = f"{mask_dir}/{image_id}.png"

        try:
            predicted_mask, iou_score = evaluate_wrestler_detection(image_path, mask_path)

            # Save the predicted mask
            output_path = f"predicted_mask_{image_id}.png"
            cv2.imwrite(output_path, predicted_mask)

            print(f"Image {image_id}: IoU = {iou_score:.4f}")
            total_iou += iou_score

        except Exception as e:
            print(f"Error processing image {image_id}: {str(e)}")
            image_count -= 1

    if image_count > 0:
        average_iou = total_iou / image_count
        print("-" * 60)
        print(f"Average IoU: {average_iou:.4f}")
        print(f"Target IoU: 0.8000")

        if average_iou >= 0.8:
            print("🎉 SUCCESS: Achieved target IoU ≥ 0.8!")
        elif average_iou >= 0.7:
            print("🔥 EXCELLENT: Very close to target!")
        elif average_iou >= 0.6:
            print("✅ GOOD: Significant improvement achieved!")
        else:
            print("❌ FAIL: Below target, further optimization needed")

        print(f"Status: {'PASS' if average_iou >= 0.8 else 'FAIL'}")
    else:
        print("Could not process any images.")


if __name__ == "__main__":
    main()