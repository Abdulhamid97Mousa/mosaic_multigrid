"""Low-level rendering primitives for grid tile drawing."""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray as ndarray


def downsample(img: ndarray, factor: int) -> ndarray:
    """Downsample an image along both dimensions by some factor."""
    assert img.shape[0] % factor == 0
    assert img.shape[1] % factor == 0
    img = img.reshape([img.shape[0] // factor, factor, img.shape[1] // factor, factor, 3])
    img = img.mean(axis=3)
    img = img.mean(axis=1)
    return img


def fill_coords(img: ndarray, fn, color) -> ndarray:
    """Fill pixels of an image with coordinates matching a filter function."""
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]
            xf = (x + 0.5) / img.shape[1]
            if fn(xf, yf):
                img[y, x] = color
    return img


def rotate_fn(fin, cx: float, cy: float, theta: float):
    """Create a rotated version of a coordinate filter function."""
    def fout(x, y):
        x = x - cx
        y = y - cy
        x2 = cx + x * math.cos(-theta) - y * math.sin(-theta)
        y2 = cy + y * math.cos(-theta) + x * math.sin(-theta)
        return fin(x2, y2)
    return fout


def point_in_line(x0: float, y0: float, x1: float, y1: float, r: float):
    """Return a function that tests if a point is within distance r of a line segment."""
    p0 = np.array([x0, y0])
    p1 = np.array([x1, y1])
    dir = p1 - p0
    dist = np.linalg.norm(dir)
    dir = dir / dist

    xmin = min(x0, x1) - r
    xmax = max(x0, x1) + r
    ymin = min(y0, y1) - r
    ymax = max(y0, y1) + r

    def fn(x, y):
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return False
        q = np.array([x, y])
        pq = q - p0
        a = np.clip(np.dot(pq, dir), 0, dist)
        p = p0 + a * dir
        return np.linalg.norm(q - p) <= r

    return fn


def point_in_circle(cx: float, cy: float, r: float):
    """Return a function that tests if a point is inside a circle."""
    def fn(x, y):
        return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r
    return fn


def point_in_rect(xmin: float, xmax: float, ymin: float, ymax: float):
    """Return a function that tests if a point is inside a rectangle."""
    def fn(x, y):
        return x >= xmin and x <= xmax and y >= ymin and y <= ymax
    return fn


def point_in_triangle(a, b, c):
    """Return a function that tests if a point is inside a triangle."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    def fn(x, y):
        v0 = c - a
        v1 = b - a
        v2 = np.array((x, y)) - a
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        return (u >= 0) and (v >= 0) and (u + v) < 1

    return fn


def highlight_img(img: ndarray, color=(255, 255, 255), alpha: float = 0.30):
    """Add highlighting overlay to an image (in-place)."""
    blend_img = img + alpha * (np.array(color, dtype=np.uint8) - img)
    blend_img = blend_img.clip(0, 255).astype(np.uint8)
    img[:, :, :] = blend_img
