import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

from visualizations import Sam2Viz

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


from skimage import morphology, feature
from skimage.feature import blob_dog
from skimage.measure import regionprops   
from skimage.color import rgb2gray

from scipy import ndimage
from scipy.ndimage import gaussian_filter, distance_transform_edt, label, center_of_mass

    
def open_and_rescale_img(path, target_w, target_h, plot=False):
    # Define the target size
    target_size = (target_w, target_h)
    
    # Load the original image
    original_image = cv2.imread(path)
    original_height, original_width = original_image.shape[:2]
    
    print(f"ORIGINAL IMAGE'S ({path}) DIMENSION: WIDTH = {original_width} --- HEIGHT = {original_height}")

    # Determine if the image is being shrunk or upscaled
    if original_width == target_size[0] and original_height == target_size[1]:
        image_resized = original_image
        
    elif original_width > target_size[0] or original_height > target_size[1]:
        # Use interpolation=cv2.INTER_AREA when shrinking
        image_resized = cv2.resize(original_image, target_size, interpolation=cv2.INTER_AREA)  # Shrinking
        was_shrunk = True
    else:
        # Use interpolation=cv2.INTER_CUBIC when upscaling
        image_resized = cv2.resize(original_image, target_size, interpolation=cv2.INTER_CUBIC)  # Upscaling
        was_shrunk = False
    
    if plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image_resized)
        plt.show()
    
    return image_resized


def cells_image_preprocessing(path = None, rescaled_img=None):
    if path is not None:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    elif rescaled_img is not None:
        # Grayscale if not already
        image = rescaled_img if len(rescaled_img.shape) == 2 else cv2.cvtColor(rescaled_img, cv2.COLOR_BGR2GRAY)

    # Step 1: Noise reduction/image smoothing with Gaussian Blur
    ## Kernel Size: The Gaussian kernel size directly impacts the level of smoothing. A larger kernel size (e.g., (7, 7) or (9, 9)) will smooth more, which might help reduce noise but can also blur small cells.
    ## Sigma Value: Adjust the sigma value to control the extent of the blurring. A lower sigma value keeps edges sharper, while a higher value enhances noise reduction.
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    # Step 2: Contrast enhancement Contrast Limited Adaptive Histogram Equalization (CLAHE). (useful for highlighting structures in images with uneven illumination)
    ## Clip Limit: The clip limit controls the contrast enhancement. Higher values (e.g., 3.0 or 4.0) increase contrast but may also amplify noise.
    ## Tile Grid Size: This parameter affects the region size where the histogram equalization is applied. A smaller tile size (e.g., (4, 4)) enhances local contrast but may over-enhance smaller regions.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(blurred)

    # Step 3: Background illumination correction
    ## A larger sigma leads to a smoother background estimation, which is ideal for removing gradual illumination changes. However, too large a sigma might result in over-smoothing, where actual cell features are mistaken for background.
    background = gaussian_filter(enhanced_image, sigma=15)
    corrected_image = enhanced_image - background
    corrected_image = np.clip(corrected_image, 0, 255)

    # Step 4: Thresholding (Otsu's) to binarize the image automatically, which helps in separating cells from the background.
    ## Binarization Threshold: While Otsu's method automatically determines the threshold, you can combine it with adaptive thresholding to address images with uneven illumination.
    ## Adaptive Thresholding: Use adaptive thresholding in combination with Otsu's method for more localized thresholding, particularly in images with variable illumination.
    _, binary_image = cv2.threshold(corrected_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 5: Morphological operations
    ## Affects how well small artifacts are removed and how well small holes are filled in the cells.
    binary_image = morphology.remove_small_objects(binary_image.astype(bool), min_size=150).astype(np.uint8)
    binary_image = morphology.remove_small_holes(binary_image.astype(bool), area_threshold=150).astype(np.uint8)

    ## 5. Erode to remove thin connections
    #kernel = np.ones((3,3), np.uint8)
    #eroded = cv2.erode(binary_image, kernel, iterations=1)
    ## 7. Dilate to restore cell size
    #binary_image = cv2.dilate(eroded, kernel, iterations=1)
    
    # Step 6: Create a mask to remove border noise
    height, width = binary_image.shape
    border_size = 3  # Adjust this value based on how much of the border you want to clean
    mask = np.ones((height, width), dtype=np.uint8) * 255
    mask[border_size:-border_size, border_size:-border_size] = 0
    # Apply the mask to remove border noise
    border_noise = cv2.bitwise_and(binary_image, mask)
    cleaned_image = binary_image - border_noise


    ## Optional: Additional morphological operation to smooth the edges
    #kernel = np.ones((3,3), np.uint8)
    #cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_CLOSE, kernel)
    #


    

    
    ## Smoothing the boundaries (optional)
    #kernel = np.ones((3, 3), np.uint8)
    #cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    ## Step 6: Final touches (normalization)
    #final_image = exposure.rescale_intensity(cleaned_image, in_range=(0, 255))
    return cleaned_image


def detect_centroids_and_boxes(preprocessed_image, plot=True):
    # Label connected components
    labeled_image, num_features = label(preprocessed_image)

    # Define area threshold
    area_threshold = 750

    # Initialize lists for centroids and bounding boxes
    individual_centroids, individual_c_boxes = [], []
    multiple_centroids, multiple_c_boxes = [], []

    # Get the dimensions of the image
    max_rows, max_cols = preprocessed_image.shape

    # Process each labeled region
    for region in regionprops(labeled_image):
        # Extract the region image
        y0, x0, y1, x1 = region.bbox
        
        # Clamp bounding box coordinates to image dimensions
        y0 = max(y0, 0)
        x0 = max(x0, 0)
        y1 = min(y1, max_rows)
        x1 = min(x1, max_cols)

        # CODE TO DETECT CENTROIDS OF TOUCHING CELLS, AND CREATE A BOX BASED ON CENTROID 
        # ABANDONED FOR BOXING STRATEGY BETTER FITTING THE CELL SHAPE        
        #if region.area > area_threshold:
        #    # This region corresponds to connected blobs
        #    # We'll use a local maximum filter to find multiple centroids
        #    from scipy.ndimage import maximum_filter
        #    from scipy.ndimage import label as scipy_label
        #    region_image = labeled_image[y0:y1, x0:x1] == region.label
        #    # Apply distance transform
        #    distance = distance_transform_edt(region_image)
        #    # Use maximum filter to find local maxima
        #    neighborhood_size = 14  # Adjust this value based on your image scale
        #    local_max = maximum_filter(distance, footprint=np.ones((neighborhood_size, neighborhood_size))) == distance
        #    # Remove edge maxima
        #    local_max[distance < 0.5 * neighborhood_size] = False
        #    # Label and find centroids of local maxima
        #    labeled_maxima, num_maxima = scipy_label(local_max)
        #    maxima_centroids = center_of_mass(local_max, labeled_maxima, range(1, num_maxima + 1))
        #    # Adjust centroids to global image coordinates
        #    for centroid in maxima_centroids:
        #        global_centroid = [centroid[0] + y0, centroid[1] + x0]
        #        # Clamp centroid coordinates to image dimensions
        #        global_centroid[0] = max(min(global_centroid[0], max_rows - 1), 0)
        #        global_centroid[1] = max(min(global_centroid[1], max_cols - 1), 0)
        #        multiple_centroids.append(global_centroid)
        #       
        #        # Create bounding box around each centroid
        #        centroid_row, centroid_col = global_centroid
        #        box_size = neighborhood_size + 25  # Adjust box size if necessary
        #        box_y0 = max(int(centroid_row - box_size / 2), 0)
        #        box_x0 = max(int(centroid_col - box_size / 2), 0)
        #        box_y1 = min(int(centroid_row + box_size / 2), max_rows)
        #        box_x1 = min(int(centroid_col + box_size / 2), max_cols)
        #       
        #        # Append in x0, y0, x1, y1 format
        #        multiple_c_boxes.append([box_x0, box_y0, box_x1, box_y1])
                
                   
        from skimage.segmentation import watershed
        from scipy import ndimage as ndi

        if region.area > area_threshold:
            # This region corresponds to connected blobs
            region_image = labeled_image[y0:y1, x0:x1] == region.label

            # Apply distance transform
            distance = ndi.distance_transform_edt(region_image)

            # Find local maxima (cell centers)
            from scipy.ndimage import maximum_filter
            neighborhood_size = 14  # Adjust this value based on your image scale
            local_max = maximum_filter(distance, footprint=np.ones((neighborhood_size, neighborhood_size))) == distance
            local_max[distance < 0.5 * neighborhood_size] = False

            # Label maxima
            markers, _ = ndi.label(local_max)

            # Apply watershed segmentation
            labels = watershed(-distance, markers, mask=region_image)

            # Process each segmented cell
            for labl in range(1, labels.max() + 1):
                cell_mask = labels == labl

                # Find bounding box of the cell
                rows, cols = np.where(cell_mask)
                if len(rows) > 0 and len(cols) > 0:
                    y_min, y_max = rows.min(), rows.max()
                    x_min, x_max = cols.min(), cols.max()

                    # Adjust to global coordinates
                    global_box = [x_min + x0, y_min + y0, x_max + x0, y_max + y0]

                    # Clamp coordinates to image dimensions
                    global_box[0] = max(global_box[0], 0)
                    global_box[1] = max(global_box[1], 0)
                    global_box[2] = min(global_box[2], max_cols - 1)
                    global_box[3] = min(global_box[3], max_rows - 1)

                    multiple_c_boxes.append(global_box)
                    
                    # Calculate centroid ensuring it's within the blob
                    y, x = np.where(cell_mask)
                    centroid = [int(y.mean()), int(x.mean())]

                    # Ensure centroid is within the blob
                    while not cell_mask[centroid[0], centroid[1]]:
                        dists = ((y - centroid[0])**2 + (x - centroid[1])**2)**0.5
                        closest_idx = dists.argmin()
                        centroid = [y[closest_idx], x[closest_idx]]

                    # Adjust to global coordinates
                    global_centroid = [centroid[0] + y0, centroid[1] + x0]

                    # Clamp centroid coordinates to image dimensions
                    global_centroid[0] = max(min(global_centroid[0], max_rows - 1), 0)
                    global_centroid[1] = max(min(global_centroid[1], max_cols - 1), 0)

                    multiple_centroids.append(global_centroid)
        
        else:
            # Calculate centroid for small regions
            centroid = list(region.centroid)

            # Clamp centroid coordinates to image dimensions
            centroid[0] = max(min(centroid[0], max_rows - 1), 0)
            centroid[1] = max(min(centroid[1], max_cols - 1), 0)

            individual_centroids.append(centroid)

            # Add bounding box for small regions, clamped to image dimensions
            individual_c_boxes.append([
                max(x0, 0), max(y0, 0),
                min(x1, max_cols), min(y1, max_rows)
            ])

    if plot:
        # Plot the results
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(preprocessed_image, cmap='gray')
        ax.set_title("Centroids and Bounding Boxes of Blobs")


        # Plot individual centroids in blue
        for c in individual_centroids:
            ax.plot(c[1], c[0], 'b+', markersize=5)

        # Plot multiple centroids from large regions in red
        for c in multiple_centroids:
            ax.plot(c[1], c[0], 'r+', markersize=5)

        import matplotlib.patches as patches

        # Plot bounding boxes of individual cells
        for bbox in  individual_c_boxes:
            x0, y0, x1, y1 = bbox
            width, height = x1 - x0, y1 - y0
            rect = patches.Rectangle((x0, y0), width, height, linewidth=1.5, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)
         
        # Plot bounding boxes of touching cells
        for bbox in  multiple_c_boxes:
            x0, y0, x1, y1 = bbox
            width, height = x1 - x0, y1 - y0
            rect = patches.Rectangle((x0, y0), width, height, linewidth=1.5, edgecolor='red', facecolor='none')
            ax.add_patch(rect)   

        plt.show()

    return (individual_centroids, individual_c_boxes), (multiple_centroids, multiple_c_boxes)






def show_point_on_img(img, fig_size, input_points, input_labels):    
    # Create figure and axis for plotting
    fig, ax = plt.subplots()
    
    # Display the image
    ax.imshow(img)
    
    # Display the cell centers
    Sam2Viz.show_points(input_points, input_labels, ax, marker='x', marker_size=30)
    
    # Set title and show plot
    ax.set_title("Detected Cell Centers")
    plt.axis('off')  # Hide axis
    plt.show()

    
    
    
    
def cell_contours_points(preprocessed_img, max_sigma=1, threshold=0):
    blobs = blob_dog(preprocessed_img, max_sigma=max_sigma, threshold=threshold)
    # Extracting the cell contours coordinates from the blobs
    cell_contour_points = blobs[:, :2]
    return cell_contour_points
    
def get_background_points(preprocessed_img, min_distance=4, threshold_abs=25):

    # Compute the distance transform
    dist_transform = cv2.distanceTransform(preprocessed_img, cv2.DIST_L2, 5)
    
    # Normalize the distance image for range = {0.0, 1.0}
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    
    # Find local maxima
    local_max = ndimage.maximum_filter(dist_transform, size=min_distance)
    mask = (dist_transform == local_max)
    mask[dist_transform < threshold_abs/255.] = 0
    
    # Find and draw the background points
    background_points = np.column_stack(np.where(mask))

    return background_points


# MUCH BETTER RESULTS IF USING BETTER RES IMG SCALING + BETTER MODEL THAN THE TINY ONE
# MUCH BETTER RESULTS IF USING BETTER RES IMG SCALING + BETTER MODEL THAN THE TINY ONE
# MUCH BETTER RESULTS IF USING BETTER RES IMG SCALING + BETTER MODEL THAN THE TINY ONE    
    
sam2_checkpoint = "segment-anything-2\checkpoints\sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
img_path = r'src\MicrosoftTeams-image_14.webp'
#img_path = r'src\truck.jpg'
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)


# Load the image and convert it to an RGB array
fff = np.array(Image.open(img_path).convert("RGB"))





rescaled_raw_img = open_and_rescale_img(img_path, 493, 493
                                        #512, 512
                                        )
preprocessed_img = cells_image_preprocessing(#path=img_path, 
                                             rescaled_img=rescaled_raw_img
                                             )
# Convert binary image back to a 3-channel grayscale image
grayscale_image = preprocessed_img * 255  # Convert binary to grayscale (0 -> 0, 1 -> 255)
image = cv2.merge([grayscale_image, grayscale_image, grayscale_image])

# Convert the single-channel grayscale image to a 3-channel grayscale image
#image = np.stack((preprocessed_img,)*3, axis=-1)



(individual_centroids, individual_c_boxes), (multiple_centroids, multiple_c_boxes) = detect_centroids_and_boxes(preprocessed_img)
len_pos = len(individual_centroids) + len(multiple_centroids)

cell_contour_points = cell_contours_points(preprocessed_img)
background_points = get_background_points(preprocessed_img)
len_neg = len(cell_contour_points) + len(background_points)

#neg_and_pos_points = np.concatenate([background_points, cell_contour_points, individual_centroids, multiple_centroids], axis=0)
#neg_and_pos_labs = [0] * len_neg + [1] * len_pos
#neg_and_pos_labels = np.array(neg_and_pos_labs)


#
#
##image = cells_image_preprocessing('src\MicrosoftTeams-image_14.webp')
fig_size=(10,10)
#input_points = np.array(neg_points)
#input_labels = np.array([0 for n in range(len(neg_points))])
#show_point_on_img(image, fig_size, points, labels)
#
#
#mask_generator = SAM2AutomaticMaskGenerator(sam2, points_per_batch=16)
#masks = mask_generator.generate(image)
#Sam2Viz.supervision_show_masks(image, masks)


considered_centroids, considered_boxes = multiple_centroids, individual_c_boxes+multiple_c_boxes  # Assuming you want to process individual blobs

print(considered_boxes)

input_points = None #np.array(considered_centroids)
pos_labs = [1] #* len(considered_centroids)
input_labels = None #np.array(pos_labs)

input_boxes = np.array(considered_boxes)




predictor = SAM2ImagePredictor(sam2)
predictor.set_image(image)

split_boxes = np.array_split(input_boxes, 2)
mask_results = []

# Print each part to see the splits
for i, split in enumerate(split_boxes):
    print(f"Split {i+1}:")
    print(split)
    print('----')
    
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):  
    for boxes in split_boxes:
        print('----', boxes, '----')
        # Inference.
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
        #sorted_ind = np.argsort(scores)[::-1]
        #masks, scores, logits = masks[sorted_ind], scores[sorted_ind], logits[sorted_ind]

        print(masks.shape)
        mask_results.extend(masks)
    

plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in mask_results:
    Sam2Viz.show_mask(mask.squeeze(0)
                      , plt.gca(), random_color=True)
#for box in input_boxes:
#    Sam2Viz.show_box(box, plt.gca())
plt.axis('off')
plt.show()

