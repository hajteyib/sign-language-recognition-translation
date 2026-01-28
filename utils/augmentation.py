import numpy as np
import torch

class LandmarkAugmentation:
    """
    Data augmentation for sign language landmarks.
    Applies spatial and temporal transformations.
    """
    def __init__(
        self,
        rotation_deg=15,
        scale_range=(0.9, 1.1),
        translation_range=0.05,
        temporal_mask_prob=0.1,
        temporal_mask_max_len=5,
        apply_prob=0.5
    ):
        """
        Args:
            rotation_deg: Max rotation angle around z-axis (degrees)
            scale_range: (min, max) scaling factor
            translation_range: Max translation as fraction of coordinate range
            temporal_mask_prob: Probability of masking a frame
            temporal_mask_max_len: Maximum consecutive frames to mask
            apply_prob: Probability of applying augmentation
        """
        self.rotation_deg = rotation_deg
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.temporal_mask_prob = temporal_mask_prob
        self.temporal_mask_max_len = temporal_mask_max_len
        self.apply_prob = apply_prob
    
    def __call__(self, landmarks_dict):
        """
        Apply augmentation to landmarks.
        
        Args:
            landmarks_dict: Dict with 'pose', 'left_hand', 'right_hand', 'face'
                           Each is numpy array (T, N, C) or torch tensor
        
        Returns:
            augmented_dict: Augmented landmarks with same structure
        """
        # Convert to numpy if torch tensor
        is_torch = isinstance(landmarks_dict['pose'], torch.Tensor)
        if is_torch:
            landmarks_dict = {k: v.numpy() for k, v in landmarks_dict.items()}
        
        # Apply augmentation with probability
        if np.random.rand() > self.apply_prob:
            if is_torch:
                return {k: torch.from_numpy(v) for k, v in landmarks_dict.items()}
            return landmarks_dict
        
        augmented = {}
        
        # Spatial augmentation (same transform for all parts)
        rotation_angle = np.random.uniform(-self.rotation_deg, self.rotation_deg)
        scale_factor = np.random.uniform(*self.scale_range)
        translation = np.random.uniform(-self.translation_range, self.translation_range, size=3)
        
        for key in ['pose', 'left_hand', 'right_hand', 'face']:
            data = landmarks_dict[key].copy()  # (T, N, C)
            
            # Spatial transformations on x, y, z coordinates
            if key == 'pose':
                # Pose has (x, y, z, visibility) - last channel is visibility
                coords = data[..., :3]  # (T, N, 3)
                visibility = data[..., 3:4]  # (T, N, 1)
                
                # Apply transformations
                coords = self._apply_spatial_transform(coords, rotation_angle, scale_factor, translation)
                
                # Recombine
                augmented[key] = np.concatenate([coords, visibility], axis=-1)
            else:
                # Hands and face have (x, y, z)
                coords = data  # (T, N, 3)
                augmented[key] = self._apply_spatial_transform(coords, rotation_angle, scale_factor, translation)
        
        # Temporal masking (applied to all parts together)
        augmented = self._apply_temporal_masking(augmented)
        
        # Convert back to torch if needed
        if is_torch:
            augmented = {k: torch.from_numpy(v.astype(np.float32)) for k, v in augmented.items()}
        
        return augmented
    
    def _apply_spatial_transform(self, coords, rotation_angle, scale_factor, translation):
        """
        Apply rotation, scaling, and translation to coordinates.
        
        Args:
            coords: (T, N, 3) array
            rotation_angle: Rotation in degrees around z-axis
            scale_factor: Scaling factor
            translation: (3,) translation vector
        
        Returns:
            transformed: (T, N, 3) array
        """
        # Rotation matrix around z-axis (yaw in camera coordinates)
        theta = np.radians(rotation_angle)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])
        
        # Apply rotation: (T, N, 3) @ (3, 3).T = (T, N, 3)
        rotated = coords @ rotation_matrix.T
        
        # Apply scaling
        scaled = rotated * scale_factor
        
        # Apply translation
        translated = scaled + translation
        
        return translated
    
    def _apply_temporal_masking(self, landmarks_dict):
        """
        Randomly mask (zero out) consecutive frames.
        
        Args:
            landmarks_dict: Dict of arrays (T, N, C)
        
        Returns:
            masked_dict: Dict with some frames masked
        """
        T = landmarks_dict['pose'].shape[0]
        
        # Decide which frames to mask
        mask = np.random.rand(T) > self.temporal_mask_prob
        
        # Optionally create consecutive masked regions
        # For simplicity, just use random individual frames for now
        
        # Apply mask to all parts
        masked = {}
        for key, data in landmarks_dict.items():
            masked_data = data.copy()
            masked_data[~mask] = 0  # Zero out masked frames
            masked[key] = masked_data
        
        return masked


class TemporalSpeedAugmentation:
    """
    Augment by changing video speed (temporal interpolation/decimation).
    """
    def __init__(self, speed_range=(0.8, 1.2), apply_prob=0.3):
        """
        Args:
            speed_range: (min, max) speed multiplier
            apply_prob: Probability of applying this augmentation
        """
        self.speed_range = speed_range
        self.apply_prob = apply_prob
    
    def __call__(self, landmarks_dict):
        """
        Change temporal speed by resampling.
        
        Args:
            landmarks_dict: Dict with arrays (T, N, C)
        
        Returns:
            resampled_dict: Dict with different T
        """
        if np.random.rand() > self.apply_prob:
            return landmarks_dict
        
        T = landmarks_dict['pose'].shape[0]
        speed_factor = np.random.uniform(*self.speed_range)
        new_T = max(1, int(T / speed_factor))
        
        # Resample indices
        old_indices = np.arange(T)
        new_indices = np.linspace(0, T - 1, new_T)
        
        resampled = {}
        for key, data in landmarks_dict.items():
            # Interpolate along time axis
            # For simplicity, use nearest neighbor
            resampled_data = data[np.round(new_indices).astype(int)]
            resampled[key] = resampled_data
        
        return resampled
