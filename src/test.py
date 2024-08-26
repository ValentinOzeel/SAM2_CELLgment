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

    
    

def cells_image_preprocessing(path):
    # Load the fluorescent cell image
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

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

    ## Smoothing the boundaries (optional)
    #kernel = np.ones((3, 3), np.uint8)
    #cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    ## Step 6: Final touches (normalization)
    #final_image = exposure.rescale_intensity(cleaned_image, in_range=(0, 255))
    return binary_image




def detect_cc(preprocessed_image, plot=True):

    # Label connected components
    labeled_image, num_features = label(preprocessed_image)

    # Define area threshold
    area_threshold = 750

    # Initialize lists for centroids
    individual_centroids = []
    multiple_centroids = []

    # Process each labeled region
    for region in regionprops(labeled_image):
        if region.area > area_threshold:
            # This region corresponds to connected blobs
            # We'll use a local maximum filter to find multiple centroids
            from scipy.ndimage import maximum_filter
            from scipy.ndimage import label as scipy_label

            # Extract the region image
            min_row, min_col, max_row, max_col = region.bbox
            region_image = labeled_image[min_row:max_row, min_col:max_col] == region.label

            # Apply distance transform
            distance = distance_transform_edt(region_image)

            # Use maximum filter to find local maxima
            neighborhood_size = 14  # Adjust this value based on your image scale
            local_max = maximum_filter(distance, footprint=np.ones((neighborhood_size, neighborhood_size))) == distance

            # Remove edge maxima
            local_max[distance < 0.5 * neighborhood_size] = False

            # Label and find centroids of local maxima
            labeled_maxima, num_maxima = scipy_label(local_max)
            maxima_centroids = center_of_mass(local_max, labeled_maxima, range(1, num_maxima + 1))

            # Adjust centroids to global image coordinates
            for centroid in maxima_centroids:
                global_centroid = (centroid[0] + min_row, centroid[1] + min_col)
                multiple_centroids.append(global_centroid)
        
        
        else:
            # Calculate centroid for small regions
            individual_centroids.append(region.centroid)

    if plot:
        # Plot the results
        plt.figure(figsize=(10, 10))
        plt.imshow(preprocessed_image, cmap='gray')
        plt.title("Centroids of Individual and Connected Blobs")

        # Plot individual centroids in blue
        for c in individual_centroids:
            plt.plot(c[1], c[0], 'b+', markersize=15, label='Individual Centroid')

        # Plot multiple centroids from large regions in red
        for c in multiple_centroids:
            plt.plot(c[1], c[0], 'r+', markersize=15, label='Multiple Centroids')


        plt.show()

    return individual_centroids, multiple_centroids




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





def detect_cell_centers(preprocessed_image):
    # Step 1: Use the Laplacian of Gaussian (LoG) method to detect blobs (potential cells)
    blobs_log = feature.blob_log(preprocessed_image, max_sigma=50, num_sigma=50, threshold=1)

    # Convert the coordinates to (x, y) and store them as centroids
    centroids = [(int(blob[1]), int(blob[0])) for blob in blobs_log]

    return centroids

    
    
sam2_checkpoint = "segment-anything-2\checkpoints\sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
img_path = r'src\MicrosoftTeams-image_14.webp'
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)


image = np.array(Image.open(img_path).convert("RGB"))
preprocessed_img = cells_image_preprocessing(img_path)

individual_centroids, multiple_centroids = detect_cc(preprocessed_img)
len_pos = len(individual_centroids) + len(multiple_centroids)

cell_contour_points = cell_contours_points(preprocessed_img)
background_points = get_background_points(preprocessed_img)
len_neg = len(cell_contour_points) + len(background_points)

#neg_and_pos_points = np.concatenate([background_points, cell_contour_points, individual_centroids, multiple_centroids], axis=0)
#neg_and_pos_labs = [0] * len_neg + [1] * len_pos
#neg_and_pos_labels = np.array(neg_and_pos_labs)


points = np.concatenate([individual_centroids, multiple_centroids], axis=0)
pos_labs = [1] * len_pos
labels = np.array(pos_labs)

#
#
##image = cells_image_preprocessing('src\MicrosoftTeams-image_14.webp')
fig_size=(10,10)
#input_points = np.array(neg_points)
#input_labels = np.array([0 for n in range(len(neg_points))])
show_point_on_img(image, fig_size, points, labels)
#
#
#mask_generator = SAM2AutomaticMaskGenerator(sam2, points_per_batch=16)
#masks = mask_generator.generate(image)
#Sam2Viz.supervision_show_masks(image, masks)


predictor = SAM2ImagePredictor(sam2)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=False
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks, scores, logits = masks[sorted_ind], scores[sorted_ind], logits[sorted_ind]

    print(masks.shape)
    Sam2Viz.show_masks(image, masks, scores, point_coords=points, input_labels=labels)






