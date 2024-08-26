

import numpy as np
np.random.seed(42)

import matplotlib.pyplot as plt
import cv2

import supervision as sv

class Sam2Viz:
    
    @staticmethod
    def supervision_show_masks(image, masks):
        mask_annotator = sv.MaskAnnotator()
        detections = sv.Detections.from_sam(masks)
        detections.class_id = [i for i in range(len(detections))]
        annotated_image = mask_annotator.annotate(image, detections)

        sv.plot_image(image=annotated_image, size=(8, 8))





    @staticmethod
    def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=None):
        """
        Display multiple masks on the image with optional points and boxes.
        """
        for i, (mask, score) in enumerate(zip(masks, scores)):
            Sam2Viz._plot_mask_with_annotations(image, mask, score, point_coords, box_coords, input_labels, borders, i)
    
    @staticmethod
    def _plot_mask_with_annotations(image, mask, score, point_coords, box_coords, input_labels, borders, index):
        """Helper function to plot a mask with optional annotations."""
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        Sam2Viz.show_mask(mask, plt.gca(), borders=borders)
        
        if point_coords is not None and input_labels is not None:
            Sam2Viz.show_points(point_coords, input_labels, plt.gca())
        
        if box_coords is not None:
            Sam2Viz.show_box(box_coords, plt.gca())
        
        #if score is not None:
            #plt.title(f"Mask {index + 1}, Score: {score:.3f}", fontsize=18)
        
        plt.axis('off')
        plt.show()
        

    @staticmethod
    def show_mask(mask, ax, random_color=False, borders=False):
        """
        Display a binary mask on a given axis with optional random color and borders.
        """
        color = Sam2Viz._generate_color(random_color)
        mask_image = Sam2Viz._apply_mask_color(mask, color)
        
        if borders:
            mask_image = Sam2Viz._draw_mask_borders(mask, mask_image)
        
        ax.imshow(mask_image)
    
    @staticmethod
    def _generate_color(random_color):
        """Generate a color for the mask, either random or fixed."""
        if random_color:
            return np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        return np.array([30/255, 144/255, 255/255, 0.6])

    @staticmethod
    def _apply_mask_color(mask, color):
        """Apply the color to the mask."""
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        return mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    @staticmethod
    def _draw_mask_borders(mask, mask_image):
        """Draw borders around the mask using contours."""
        mask = (mask > 0).astype(np.uint8)  # Convert mask to binary (0 or 1) and then to uint8
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        return cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=1)
    
    @staticmethod
    def show_points(coords, labels, ax, marker='X', marker_size=50):
        """
        Display positive and negative points on the given axis.
        """
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        
        Sam2Viz._plot_points(pos_points, ax, 'green', marker, marker_size)
        Sam2Viz._plot_points(neg_points, ax, 'red', marker, marker_size)
    
    @staticmethod
    def _plot_points(points, ax, color, marker, marker_size):
        """Helper function to plot points with a specific color and marker size."""
        ax.scatter(points[:, 1], points[:, 0], color=color, marker=marker, s=marker_size,
                   edgecolor='white', linewidth=1)

    @staticmethod
    def show_box(box, ax):
        """
        Display a bounding box on the given axis.
        """
        x0, y0 = box[:2]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    

