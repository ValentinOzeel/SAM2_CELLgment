

import numpy as np
np.random.seed(42)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
        
import cv2
from typing import List

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
    def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            Sam2Viz.show_mask(mask, plt.gca(), borders=borders)
            if point_coords is not None:
                assert input_labels is not None
                Sam2Viz.show_points(point_coords, input_labels, plt.gca())
            if box_coords is not None:
                # boxes
                Sam2Viz.show_box(box_coords, plt.gca())
            if len(scores) > 1:
                plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()


    @staticmethod
    def show_mask(mask, ax, random_color=False, borders=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if borders:
            import cv2
            contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
        ax.imshow(mask_image)

    @staticmethod
    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

    @staticmethod
    def show_box(box, ax):
        print('BOXXXXXXXX')
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    


