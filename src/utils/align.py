import cv2
import numpy as np

def align_face(img, landmarks, output_size=(256, 256)):
    """
    Aligns a face using reference landmarks (eyes + nose)
    
    Args:
        img (np.array): original image (H,W,3)
        landmarks (np.array): 106 landmarks normalized (0-1), shape (106,2)
        output_size (tuple): (width, height) of aligned output

    Returns:
        aligned_face (np.array): aligned cropped face
        M (np.array): affine transformation matrix
    """
    # Denormalize landmarks to image coordinates
    h, w, _ = img.shape
    lm = np.array([[x * w, y * h] for x, y in landmarks])

    # Use left eye, right eye, nose tip as reference points
    # Landmark indices according to WFLW:
    # left eye: average of 60-67
    # right eye: average of 68-75
    # nose tip: 30

    left_eye = np.mean(lm[60:68], axis=0)
    right_eye = np.mean(lm[68:76], axis=0)
    nose_tip = lm[30]

    src_pts = np.array([left_eye, right_eye, nose_tip], dtype=np.float32)

    # Destination points for aligned output
    dst_pts = np.array([
        [output_size[0]*0.3, output_size[1]*0.35],
        [output_size[0]*0.7, output_size[1]*0.35],
        [output_size[0]*0.5, output_size[1]*0.6]
    ], dtype=np.float32)

    # Compute affine transform
    M = cv2.getAffineTransform(src_pts, dst_pts)
    aligned_face = cv2.warpAffine(img, M, output_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return aligned_face, M