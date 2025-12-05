#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Video Processing Utility for Landmark-based Transformations

This module provides utilities for processing video frames with facial landmarks,
including landmark interpolation, affine transformations, and patch cropping.

Key Features:
- Landmark interpolation for missing frames
- Affine transformation of video frames
- Patch cropping based on facial landmarks
- Grayscale conversion option

Dependencies:
- OpenCV (cv2)
- NumPy
- scikit-image
"""

import os
import cv2
import numpy as np
from skimage import transform as tf


def linear_interpolate(landmarks, start_idx, stop_idx):
    """
    Linearly interpolate landmarks between two frames.

    Args:
        landmarks (list): List of landmarks for each frame
        start_idx (int): Starting frame index
        stop_idx (int): Ending frame index

    Returns:
        list: Updated landmarks with interpolated values
    """
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks

    # Interpolate landmarks between start and stop indices
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = (
            start_landmarks + idx / float(stop_idx - start_idx) * delta
        )
    return landmarks


def cut_patch(img, landmarks, height, width, threshold=5):
    """
    Cut a patch from the image centered around facial landmarks.

    Args:
        img (numpy.ndarray): Input image
        landmarks (numpy.ndarray): Facial landmarks
        height (int): Half-height of the patch
        width (int): Half-width of the patch
        threshold (int, optional): Tolerance for landmark positioning. Defaults to 5.

    Returns:
        numpy.ndarray: Cropped image patch

    Raises:
        OverflowError: If landmarks are too far from image center
    """
    # Calculate the center point of landmarks
    center_x, center_y = np.mean(landmarks, axis=0)

    # Validate landmark positioning
    if abs(center_y - img.shape[0] / 2) > height + threshold:
        raise OverflowError("Too much bias in height")
    if abs(center_x - img.shape[1] / 2) > width + threshold:
        raise OverflowError("Too much bias in width")

    # Calculate bounding box coordinates with edge clipping
    y_min = int(round(np.clip(center_y - height, 0, img.shape[0])))
    y_max = int(round(np.clip(center_y + height, 0, img.shape[0])))
    x_min = int(round(np.clip(center_x - width, 0, img.shape[1])))
    x_max = int(round(np.clip(center_x + width, 0, img.shape[1])))

    # Extract and return the patch
    return np.copy(img[y_min:y_max, x_min:x_max])


class VideoProcessor:
    """
    A comprehensive video processing class for landmark-based transformations.

    Handles:
    - Landmark interpolation
    - Affine transformations
    - Patch cropping
    """

    def __init__(
        self,
        mean_face_path="20words_mean_face.npy",
        crop_width=96,
        crop_height=96,
        start_idx=48,
        stop_idx=68,
        window_margin=12,
        convert_gray=True,
        do_crop=True,
    ):
        """
        Initialize the VideoProcessor.

        Args:
            mean_face_path (str): Path to mean face landmarks
            crop_width (int): Width of cropped patch
            crop_height (int): Height of cropped patch
            start_idx (int): Starting index for landmark subset
            stop_idx (int): Ending index for landmark subset
            window_margin (int): Temporal smoothing window size
            convert_gray (bool): Convert frames to grayscale
            do_crop (bool): Whether to crop patches after transformation
        """
        # Load reference landmarks
        self.reference = np.load(
            os.path.join(os.path.dirname(__file__), mean_face_path)
        )
        
        # Configuration parameters
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.start_idx = start_idx
        self.stop_idx = stop_idx
        self.window_margin = window_margin
        self.convert_gray = convert_gray
        self.do_crop = do_crop

    def __call__(self, video, landmarks):
        """
        Process video with facial landmarks.

        Args:
            video (list): List of video frames
            landmarks (list): Facial landmarks for each frame

        Returns:
            numpy.ndarray: Processed video sequence, or None if processing fails
        """
        # Interpolate missing landmarks
        preprocessed_landmarks = self.interpolate_landmarks(landmarks)
        
        # Validate preprocessing results
        if (
            not preprocessed_landmarks
            or len(preprocessed_landmarks) < self.window_margin
        ):
            return None

        # Process and transform video
        sequence = self.process_sequence(video, preprocessed_landmarks)
        assert sequence is not None, "Processing yielded an empty sequence."
        
        return sequence
    
    def process_sequence(self, video, landmarks):
        """
        Crop and transform patches for each video frame.

        Args:
            video (list): List of video frames
            landmarks (list): Preprocessed facial landmarks

        Returns:
            numpy.ndarray: Sequence of transformed patches
        """
        sequence = []
        for frame_idx, frame in enumerate(video):
            # Calculate temporal smoothing window
            window_margin = min(
                self.window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx
            )
            
            # Smooth landmarks temporally
            smoothed_landmarks = np.mean(
                [
                    landmarks[x]
                    for x in range(
                        frame_idx - window_margin, frame_idx + window_margin + 1
                    )
                ],
                axis=0,
            )
            smoothed_landmarks += landmarks[frame_idx].mean(
                axis=0
            ) - smoothed_landmarks.mean(axis=0)
            
            # Apply affine transformation
            transformed_frame, transformed_landmarks = self.affine_transform(
                frame, smoothed_landmarks, self.reference, grayscale=self.convert_gray
            )
            
            # Optional patch cropping
            if self.do_crop:
                try:
                    transformed_frame = cut_patch(
                        transformed_frame,
                        transformed_landmarks[self.start_idx : self.stop_idx],
                        self.crop_height // 2,
                        self.crop_width // 2,
                    )
                except OverflowError:
                    # If patch cutting fails, either return None or the whole transformed frame
                    return None
            
            sequence.append(transformed_frame)
        
        return np.array(sequence)

    def interpolate_landmarks(self, landmarks):
        """
        Interpolate missing landmarks across frames.

        Args:
            landmarks (list): Facial landmarks for each frame

        Returns:
            list: Interpolated landmarks, or None if no valid landmarks
        """
        # Find frames with valid landmarks
        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        if not valid_frames_idx:
            return None

        # Interpolate between frames with missing landmarks
        for idx in range(1, len(valid_frames_idx)):
            if valid_frames_idx[idx] - valid_frames_idx[idx - 1] > 1:
                landmarks = linear_interpolate(
                    landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx]
                )

        # Recheck valid frames after interpolation
        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        # Handle edge cases: fill start and end with nearest valid landmarks
        if valid_frames_idx:
            landmarks[: valid_frames_idx[0]] = [
                landmarks[valid_frames_idx[0]]
            ] * valid_frames_idx[0]
            landmarks[valid_frames_idx[-1] :] = [landmarks[valid_frames_idx[-1]]] * (
                len(landmarks) - valid_frames_idx[-1]
            )

        assert all(lm is not None for lm in landmarks), "Not every frame has a landmark"

        return landmarks

    def affine_transform(
        self,
        frame,
        landmarks,
        reference,
        grayscale=True,
        target_size=(256, 256),
        reference_size=(256, 256),
        stable_points=(28, 33, 36, 39, 42, 45, 48, 54),
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        border_value=0,
    ):
        """
        Apply affine transformation to a frame based on facial landmarks.

        Args:
            frame (numpy.ndarray): Input video frame
            landmarks (list): Facial landmarks
            reference (numpy.ndarray): Reference landmarks
            grayscale (bool): Convert frame to grayscale
            target_size (tuple): Output frame size
            reference_size (tuple): Reference frame size
            stable_points (tuple): Landmark indices used for stable transformation
            interpolation (int): OpenCV interpolation method
            border_mode (int): OpenCV border mode
            border_value (int): Border value for padding

        Returns:
            tuple: Transformed frame and transformed landmarks
        """
        # Convert to grayscale if required
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Get stable reference points
        stable_reference = self.get_stable_reference(
            reference, stable_points, reference_size, target_size
        )

        # Estimate affine transform
        transform = self.estimate_affine_transform(
            landmarks, stable_points, stable_reference
        )

        # Apply affine transform
        transformed_frame, transformed_landmarks = self.apply_affine_transform(
            frame,
            landmarks,
            transform,
            target_size,
            interpolation,
            border_mode,
            border_value,
        )

        return transformed_frame, transformed_landmarks

    def get_stable_reference(
        self, reference, stable_points, reference_size, target_size
    ):
        """
        Adjust reference points for stable transformation.

        Args:
            reference (numpy.ndarray): Reference landmarks
            stable_points (tuple): Landmark indices for stable points
            reference_size (tuple): Reference frame size
            target_size (tuple): Target frame size

        Returns:
            numpy.ndarray: Adjusted stable reference points
        """
        stable_reference = np.vstack([reference[x] for x in stable_points])
        stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
        stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0
        return stable_reference

    def estimate_affine_transform(self, landmarks, stable_points, stable_reference):
        """
        Estimate affine transformation matrix.

        Args:
            landmarks (list): Input facial landmarks
            stable_points (tuple): Landmark indices for stable points
            stable_reference (numpy.ndarray): Stable reference points

        Returns:
            numpy.ndarray: Affine transformation matrix
        """
        return cv2.estimateAffinePartial2D(
            np.vstack([landmarks[x] for x in stable_points]),
            stable_reference,
            method=cv2.LMEDS,
        )[0]

    def apply_affine_transform(
        self,
        frame,
        landmarks,
        transform,
        target_size,
        interpolation,
        border_mode,
        border_value,
    ):
        """
        Apply the estimated affine transformation to the frame and landmarks.

        Args:
            frame (numpy.ndarray): Input video frame
            landmarks (list): Facial landmarks
            transform (numpy.ndarray): Affine transformation matrix
            target_size (tuple): Output frame size
            interpolation (int): OpenCV interpolation method
            border_mode (int): OpenCV border mode
            border_value (int): Border value for padding

        Returns:
            tuple: Transformed frame and transformed landmarks
        """
        # Apply affine transform to frame
        transformed_frame = cv2.warpAffine(
            frame,
            transform,
            dsize=(target_size[0], target_size[1]),
            flags=interpolation,
            borderMode=border_mode,
            borderValue=border_value,
        )

        # Transform landmarks
        transformed_landmarks = (
            np.matmul(landmarks, transform[:, :2].transpose())
            + transform[:, 2].transpose()
        )

        return transformed_frame, transformed_landmarks