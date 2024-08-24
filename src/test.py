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


from skimage import measure, morphology, exposure, feature
from skimage.feature import blob_dog
from skimage.color import rgb2gray

from scipy import ndimage
from scipy.ndimage import gaussian_filter


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
    _, binary_image = cv2.threshold(corrected_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 5: Morphological operations
    ## Affects how well small artifacts are removed and how well small holes are filled in the cells.
    binary_image = morphology.remove_small_objects(binary_image.astype(bool), min_size=150).astype(np.uint8)
    binary_image = morphology.remove_small_holes(binary_image.astype(bool), area_threshold=100).astype(np.uint8)

    # Smoothing the boundaries (optional)
    kernel = np.ones((3, 3), np.uint8)
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # Step 6: Final touches (normalization)
    final_image = exposure.rescale_intensity(cleaned_image, in_range=(0, 255))
    return final_image


def centroids_detection_with_contour_detection(path, plot=True):
    import imutils
    
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    preprocessed_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # List to hold the cell centers
    cell_centers = []
    
    # loop over the contours
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cell_centers.append((cX, cY))


    if plot:
        # Step 5: Visualize detected centroids
        plt.imshow(preprocessed_image, cmap='gray')
        plt.scatter([x for (x, y) in cell_centers], [y for (x, y) in cell_centers], c='r', s=30, marker='x')
        plt.title("Detected Cell Centers")
        plt.show()
    
    return cell_centers



def centroids_detection_with_connected_components(preprocessed_image):
    # Step 1: Label connected regions in the binary image
    labeled_image, num_labels = measure.label(preprocessed_image, return_num=True)

    # Step 2: Remove small objects to clean up the labels
    cleaned_labels = morphology.remove_small_objects(labeled_image, min_size=100)

    # Step 3: Calculate properties of labeled regions
    props = measure.regionprops(cleaned_labels)

    # Step 4: Extract centroids of the labeled regions
    centroids = np.array([prop.centroid for prop in props])

    # Optional: Filter centroids near the image borders
    #margin = 10  # Example margin from the image border
    #image_shape = preprocessed_image.shape
    #valid_centroids = [c for c in centroids if margin < c[0] < image_shape[0] - margin and margin < c[1] < image_shape[1] - margin]
    #valid_centroids = np.array(valid_centroids)

    # Step 5: Visualize detected centroids
    plt.imshow(preprocessed_image, cmap='gray')
    plt.scatter(centroids[:, 1], centroids[:, 0], c='r', s=30, marker='x')
    plt.title("Detected Cell Centers")
    plt.show()

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
img_path = 'src\MicrosoftTeams-image_14.webp'
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)


image = np.array(Image.open(img_path).convert("RGB"))
preprocessed_img = cells_image_preprocessing(img_path)

centroids = centroids_detection_with_contour_detection(img_path)
# Plotting with matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(preprocessed_img)

# Overlay centroids
for centroid in centroids:
    plt.plot(centroid[1], centroid[0], 'ro', markersize=5)  # red circles

plt.title('Detected Cell Centers')
plt.axis('off')  # Hide axes for better visualization
plt.show()



