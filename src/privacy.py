import cv2

class Masker:
    """
    Applies privacy masking to faces or regions of the frame.
    For now this just blurs the whole frame if enabled.
    """
    def __init__(self, cfg):
        self.enabled = bool(cfg.get("video", {}).get("face_mask", False))

    def blur_faces(self, frame):
        if not self.enabled:
            return frame
        # Simple global blur (you can later replace with real face detection)
        return cv2.GaussianBlur(frame, (15, 15), 0)


class InOptOut:
    """
    Handles an 'opt-out' region of the screen where tracking is disabled.
    Region coordinates are normalized [x0, y0, x1, y1].
    """
    def __init__(self, cfg):
        self.rect = cfg.get("video", {}).get("opt_out_rect", None)

    def in_region(self, cx, cy):
        """Return True if a point (cx, cy) lies inside the opt-out rectangle."""
        if not self.rect:
            return False
        x0, y0, x1, y1 = self.rect
        return x0 <= cx <= x1 and y0 <= cy <= y1
