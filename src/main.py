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
from skimage.segmentation import watershed
from skimage.measure import regionprops  
from skimage.feature import blob_dog
from skimage.color import rgb2gray

from scipy.ndimage import gaussian_filter, distance_transform_edt, label, center_of_mass, maximum_filter

from access_config import get_config_dict

config = get_config_dict()
# PREPROCESSING
GAUSSIAN_BLUR_KERNEL_SIZE = tuple(config['PREPROCESSING']['gaussian_blur_kernel_size'])
GAUSSIAN_BLUR_SIGMA = config['PREPROCESSING']['gaussian_blur_sigma']
CLAHE_CLIP_LIMIT = config['PREPROCESSING']['clahe_clip_limit']
CLAHE_TILE_GRID_SIZE = tuple(config['PREPROCESSING']['clahe_tile_grid_size'])
GAUSSIAN_FILTER_SIGMA = config['PREPROCESSING']['gaussian_filter_sigma']
REMOVE_SMALL_OBJETCS_MIN_SIZE = config['PREPROCESSING']['remove_small_objects_min_size']
REMOVE_SMALL_HOLES_AREA_THRESHOLD = config['PREPROCESSING']['remove_small_holes_area_threshold']
BORDER_SIZE_NOISE_REMOVAL = config['PREPROCESSING']['border_size_noise_removal']
# CENTROIDS_POINTS_AND_BBOXES
AREA_THRESHOLD_TOUCHING_CELLS = config['CENTROIDS_POINTS_AND_BBOXES']['area_threshold_touching_cells']
TOUCHING_CELLS_NEIGHBORHOOD_SIZE = config['CENTROIDS_POINTS_AND_BBOXES']['touching_cells_neighborhood_size']
# CONTOUR_AND_BACKGROUND_POINTS
GET_CELL_CONTOUR_POINTS_MAX_SIGMA = config['CONTOUR_AND_BACKGROUND_POINTS']['get_cell_contour_points_max_sigma']
GET_CELL_CONTOUR_POINTS_THRESHOLD = config['CONTOUR_AND_BACKGROUND_POINTS']['get_cell_contour_points_threshold']
GET_BACKGROUND_POINTS_MIN_DISTANCE = config['CONTOUR_AND_BACKGROUND_POINTS']['get_background_points_min_distance']
GET_BACKGROUND_POINTS_THRESHOLD_ABS = config['CONTOUR_AND_BACKGROUND_POINTS']['get_background_points_threshold_abs']

  

class CellGment():

    @staticmethod
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
            Sam2Viz.show_image(image_resized)

        return image_resized


    
    @staticmethod
    def cells_image_preprocessing(path=None, image=None, plot=False):
        # Check input, path or rescaled_img should be passed
        assert path is not None or image is not None, "Eigther 'path' or 'image' kwargs should be passed"
        # Process input
        if path is not None:
            raw_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            # Grayscale if not already
            raw_image = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 1: Noise reduction/image smoothing with Gaussian Blur
        ## Kernel Size: The Gaussian kernel size directly impacts the level of smoothing. A larger kernel size (e.g., (7, 7) or (9, 9)) will smooth more, which might help reduce noise but can also blur small cells.
        ## Sigma Value: Adjust the sigma value to control the extent of the blurring. A lower sigma value keeps edges sharper, while a higher value enhances noise reduction.
        blurred = cv2.GaussianBlur(raw_image, GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_SIGMA)

        # Step 2: Contrast enhancement Contrast Limited Adaptive Histogram Equalization (CLAHE). (useful for highlighting structures in images with uneven illumination)
        ## Clip Limit: The clip limit controls the contrast enhancement. Higher values (e.g., 3.0 or 4.0) increase contrast but may also amplify noise.
        ## Tile Grid Size: This parameter affects the region size where the histogram equalization is applied. A smaller tile size (e.g., (4, 4)) enhances local contrast but may over-enhance smaller regions.
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
        enhanced_image = clahe.apply(blurred)

        # Step 3: Background illumination correction
        ## A larger sigma leads to a smoother background estimation, which is ideal for removing gradual illumination changes. However, too large a sigma might result in over-smoothing, where actual cell features are mistaken for background.
        background = gaussian_filter(enhanced_image, sigma=GAUSSIAN_FILTER_SIGMA)
        corrected_image = enhanced_image - background
        corrected_image = np.clip(corrected_image, 0, 255)

        # Step 4: Thresholding (Otsu's) to binarize the image automatically, which helps in separating cells from the background.
        ## Binarization Threshold: While Otsu's method automatically determines the threshold, you can combine it with adaptive thresholding to address images with uneven illumination.
        ## Adaptive Thresholding: Use adaptive thresholding in combination with Otsu's method for more localized thresholding, particularly in images with variable illumination.
        _, binary_image = cv2.threshold(corrected_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Step 5: Morphological operations
        ## Affects how well small artifacts are removed and how well small holes are filled in the cells.
        binary_image = morphology.remove_small_objects(binary_image.astype(bool), min_size=REMOVE_SMALL_OBJETCS_MIN_SIZE).astype(np.uint8)
        binary_image = morphology.remove_small_holes(binary_image.astype(bool), area_threshold=REMOVE_SMALL_HOLES_AREA_THRESHOLD).astype(np.uint8)

        ## 5.bis. Erode to remove thin connections and Dilate to restore cell size
        #kernel = np.ones((3,3), np.uint8)
        #eroded = cv2.erode(binary_image, kernel, iterations=1)
        #binary_image = cv2.dilate(eroded, kernel, iterations=1)

        # Step 6: Create a mask to remove border noise
        height, width = binary_image.shape
        border_size = BORDER_SIZE_NOISE_REMOVAL  # Adjust this value based on how much of the border you want to clean
        mask = np.ones((height, width), dtype=np.uint8) * 255
        mask[border_size:-border_size, border_size:-border_size] = 0
        # Apply the mask to remove border noise
        border_noise = cv2.bitwise_and(binary_image, mask)
        cleaned_binary_image = binary_image - border_noise

        if plot:
            Sam2Viz.show_image(cleaned_binary_image, gray=True)
            
        return cleaned_binary_image

    @staticmethod
    def get_cell_centroids_and_bboxes(preprocessed_image, plot=True):
        # Label connected components
        labeled_image, num_features = label(preprocessed_image)

        # Initialize lists for centroids and bounding boxes
        individual_c_centroids, individual_c_bboxes = [], []
        touching_c_centroids, touching_c_bboxes = [], []

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
            #if region.area > AREA_THRESHOLD_TOUCHING_CELLS:
            #    # This region corresponds to connected blobs
            #    # We'll use a local maximum filter to find multiple centroids
            #    from scipy.ndimage import maximum_filter
            #    from scipy.ndimage import label as scipy_label
            #    region_image = labeled_image[y0:y1, x0:x1] == region.label
            #    # Apply distance transform
            #    distance = distance_transform_edt(region_image)
            #    # Use maximum filter to find local maxima
            #    neighborhood_size = TOUCHING_CELLS_NEIGHBORHOOD_SIZE  # Adjust this value based on your image scale
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
            #        touching_c_centroids.append(global_centroid)
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
            #        touching_c_bboxes.append([box_x0, box_y0, box_x1, box_y1])

            # If the area is sup to the threshold for considering touching cells
            if region.area > AREA_THRESHOLD_TOUCHING_CELLS:
                
                # This region corresponds to connected blobs
                region_image = labeled_image[y0:y1, x0:x1] == region.label

                # Apply distance transform
                distance = distance_transform_edt(region_image)

                # Find local maxima (cell centers)
                neighborhood_size = TOUCHING_CELLS_NEIGHBORHOOD_SIZE  # Adjust this value based on your image scale
                local_max = maximum_filter(distance, footprint=np.ones((neighborhood_size, neighborhood_size))) == distance
                local_max[distance < 0.5 * neighborhood_size] = False

                # Label maxima
                markers, _ = label(local_max)

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

                        touching_c_bboxes.append(global_box)

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

                        touching_c_centroids.append(global_centroid)

            else:
                # Calculate centroid for small regions
                centroid = list(region.centroid)

                # Clamp centroid coordinates to image dimensions
                centroid[0] = max(min(centroid[0], max_rows - 1), 0)
                centroid[1] = max(min(centroid[1], max_cols - 1), 0)

                individual_c_centroids.append(centroid)

                # Add bounding box for small regions, clamped to image dimensions
                individual_c_bboxes.append([
                    max(x0, 0), max(y0, 0),
                    min(x1, max_cols), min(y1, max_rows)
                ])

        if plot:
            Sam2Viz.show_cell_centroids_and_bboxes(
                preprocessed_image,
                indiv_c_centroids=individual_c_centroids, indiv_c_bboxes=individual_c_bboxes,
                touching_c_centroids=touching_c_centroids, touching_c_bboxes=touching_c_bboxes
            )
                                          
        return (individual_c_centroids, individual_c_bboxes), (touching_c_centroids, touching_c_bboxes)

    @staticmethod
    def get_cell_contours_points(preprocessed_img, 
                                 max_sigma=GET_CELL_CONTOUR_POINTS_MAX_SIGMA, threshold=GET_CELL_CONTOUR_POINTS_THRESHOLD):
        blobs = blob_dog(preprocessed_img, max_sigma=max_sigma, threshold=threshold)
        # Extracting the cell contours coordinates from the blobs
        cell_contour_points = blobs[:, :2]
        return cell_contour_points

    @staticmethod
    def get_background_points(preprocessed_img, 
                              min_distance=GET_BACKGROUND_POINTS_MIN_DISTANCE, threshold_abs=GET_BACKGROUND_POINTS_THRESHOLD_ABS):

        # Compute the distance transform
        dist_transform = cv2.distanceTransform(preprocessed_img, cv2.DIST_L2, 5)

        # Normalize the distance image for range = {0.0, 1.0}
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

        # Find local maxima
        local_max = maximum_filter(dist_transform, size=min_distance)
        mask = (dist_transform == local_max)
        mask[dist_transform < threshold_abs/255.] = 0

        # Find and draw the background points
        background_points = np.column_stack(np.where(mask))

        return background_points


    @staticmethod
    def sam2_automasks(sam2, img, points_per_batch=16):
        mask_generator = SAM2AutomaticMaskGenerator(sam2, points_per_batch=points_per_batch)
        return mask_generator.generate(img)
    

    @staticmethod
    def sam2_predictormasks(sam2, img, point_coords=None, point_labels=None, box=None, batched_box=None, multimask_output=False):
        def inference(points=None, labels=None, bboxes=None, multi_output=False):
            masks, scores, logits = predictor.predict(
                point_coords=points,
                point_labels=labels, 
                box=bboxes,
                multimask_output=multi_output,
            )
            return masks, scores, logits
        
        def sort_multi_ouput(masks, scores, logits):
            sorted_ind = np.argsort(scores)[::-1]
            return masks[sorted_ind], scores[sorted_ind], logits[sorted_ind]

        # Set up predictor   
        predictor = SAM2ImagePredictor(sam2)
        predictor.set_image(img)

        # SAM2 Inference
        mask_results, score_results, logit_results = [], [], []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16): 
            if batched_box:
                for bboxes in batched_box:
                    masks, scores, logits = inference(
                        bboxes=bboxes
                        )
                    ## Sort if multimask_output=True
                    if multimask_output:
                        masks, scores, logits = sort_multi_ouput(masks, scores, logits)
                        
                    mask_results.extend(masks)
                    score_results.extend(scores)
                    logit_results.extend(logits)
            
            else:
                mask_results, score_results, logit_results = inference(
                    points=point_coords,
                    labels=point_labels,
                    bboxes=box,
                    multi_output=multimask_output
                )
                ## Sort if multimask_output=True
                if multimask_output:
                    mask_results, score_results, logit_results = sort_multi_ouput(mask_results, score_results, logit_results)
            
        return mask_results, score_results, logit_results
    
    
if __name__ == "__main__":
    sam2_checkpoint = "segment-anything-2\checkpoints\sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    img_path = r'src\22636-left.png'
    img_for_viz = np.array(Image.open(img_path).convert("RGB"))
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

    ## Load and rescale the image
    #rescaled_raw_img = CellGment.open_and_rescale_img(img_path, 493, 493)

    ## Load and preprocess the image to remove noise and separate cells from background
    preprocessed_img = CellGment.cells_image_preprocessing(
        path=img_path, 
        #image=rescaled_raw_img
        plot=False
    )

    ## Convert binary image back to a 3-channel grayscale image
    grayscale_image = preprocessed_img * 255  # Convert binary to grayscale (0 -> 0, 1 -> 255)
    three_channels_preprocessed_img = cv2.merge([grayscale_image, grayscale_image, grayscale_image])

    ## Get single cell centroid and bounding boxes (distinguish touching cells)
    (individual_c_centroids, individual_c_bboxes), (touching_c_centroids, touching_c_bboxes) = CellGment.get_cell_centroids_and_bboxes(preprocessed_img)

    ## Get cell contour points and some background points (to be used as negative points)
    #cell_contour_points = CellGment.get_cell_contours_points(preprocessed_img)
    #background_points = CellGment.get_background_points(preprocessed_img)

    considered_centroids, considered_boxes = individual_c_centroids+touching_c_centroids, individual_c_bboxes+touching_c_bboxes  # Assuming you want to process individual blobs
    input_points = np.array(considered_centroids)
    pos_labs = [1] * len(considered_centroids)
    input_labels = np.array(pos_labs)
    input_boxes = np.array(considered_boxes)



    ##### Generate automatic masks with SAM2 #####
    auto_masks = CellGment.sam2_automasks(sam2, three_channels_preprocessed_img)
    # Display masks on img
    Sam2Viz.supervision_show_masks(img_for_viz, auto_masks)


    # Batch the input boxes (for gpu memory poor) 
    split_boxes = np.array_split(input_boxes, 2)
    
    ##### Predict masks with SAM2ImagePredictor + boxes/prompts #####

    masks, scores, logits = CellGment.sam2_predictormasks(
        sam2, three_channels_preprocessed_img, 
        point_coords=None, point_labels=None, box=None, batched_box=split_boxes, 
        multimask_output=False
    )

            
    # Visualize masks
    plt.figure(figsize=(10, 10))
    plt.imshow(three_channels_preprocessed_img)
    for mask in masks:
        Sam2Viz.show_mask(mask.squeeze(0),
                          plt.gca(), 
                          random_color=True, borders=True)
    for box in input_boxes:
        Sam2Viz.show_box(box, plt.gca())
    plt.axis('off')
    plt.show()