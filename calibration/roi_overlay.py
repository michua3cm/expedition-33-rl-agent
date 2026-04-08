"""ROI boundary visualiser for the calibration overlay.

Draws a dashed rectangle for every target that has an ``"roi"`` key defined
in its config, converting the fractional ``(x_frac, y_frac, w_frac, h_frac)``
values to absolute pixel coordinates before passing them to the overlay.

Usage::

    from calibration.roi_overlay import draw_roi_overlays

    if show_roi:
        draw_roi_overlays(overlay, TARGETS, frame_w, frame_h, off_x, off_y)
"""

from __future__ import annotations

from overlay_ui import OverlayWindow


def draw_roi_overlays(
    overlay: OverlayWindow,
    targets: dict,
    frame_w: int,
    frame_h: int,
    off_x: int,
    off_y: int,
) -> None:
    """Draw dashed ROI boundary rectangles for all targets with an ``"roi"`` field.

    Args:
        overlay:  The active :class:`OverlayWindow` instance.
        targets:  Target config dict (same shape as ``TARGETS`` in config.py).
        frame_w:  Width of the captured frame in pixels.
        frame_h:  Height of the captured frame in pixels.
        off_x:    Horizontal pixel offset of the capture region on the monitor
                  (``monitor_config["left"]``).  Added so the rectangles are
                  positioned correctly on the full-screen overlay.
        off_y:    Vertical pixel offset of the capture region (``monitor_config["top"]``).
    """
    for label, cfg in targets.items():
        roi = cfg.get("roi")
        if roi is None:
            continue
        x = int(roi[0] * frame_w) + off_x
        y = int(roi[1] * frame_h) + off_y
        w = max(1, int(roi[2] * frame_w))
        h = max(1, int(roi[3] * frame_h))
        color: str = cfg.get("color", "white")
        overlay.draw_roi_rect(x, y, w, h, color, label)


def roi_to_pixels(
    roi: tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
    off_x: int = 0,
    off_y: int = 0,
) -> tuple[int, int, int, int]:
    """Convert a fractional ROI to absolute pixel coordinates.

    Args:
        roi:     ``(x_frac, y_frac, w_frac, h_frac)`` fractions.
        frame_w: Capture frame width in pixels.
        frame_h: Capture frame height in pixels.
        off_x:   Optional monitor x offset to add.
        off_y:   Optional monitor y offset to add.

    Returns:
        ``(x, y, w, h)`` in absolute pixels.
    """
    x = int(roi[0] * frame_w) + off_x
    y = int(roi[1] * frame_h) + off_y
    w = max(1, int(roi[2] * frame_w))
    h = max(1, int(roi[3] * frame_h))
    return x, y, w, h
