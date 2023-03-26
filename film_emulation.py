import os
import cv2
import numpy as np
import noise

def load_lut(file_path, grid_size=8):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]

    lut = np.zeros((grid_size, grid_size, grid_size, 3), dtype=np.float32)
    index = 0
    for line in lines:
        if 'LUT_3D_SIZE' in line:
            grid_size = int(line.split()[-1])
            lut = np.zeros((grid_size, grid_size, grid_size, 3), dtype=np.float32)
        elif 'LUT_3D_INPUT_RANGE' not in line:
            i, j, k = np.unravel_index(index, (grid_size, grid_size, grid_size))
            r, g, b = map(float, line.split())
            lut[i, j, k] = [r, g, b]
            index += 1

    return lut

def apply_lut(image, lut):
    return cv2.LUT(image, lut)

def apply_halation(image, threshold=220, blur_radius=30, intensity=0.5):
    image_float = image.astype(np.float32)
    ycrcb_image = cv2.cvtColor(image_float, cv2.COLOR_BGR2YCrCb)

    brightness_mask = ycrcb_image[..., 0] > threshold
    chroma_diff = np.abs(ycrcb_image[..., 1] - ycrcb_image[..., 2])
    chroma_mask = chroma_diff > 15

    combined_mask = (brightness_mask & chroma_mask).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    blurred_image = cv2.GaussianBlur(image_float, (blur_radius, blur_radius), 0)
    combined_mask = combined_mask[:, :, np.newaxis].astype(np.float32)
    halation_image = image_float * (1 - combined_mask) + blurred_image * combined_mask

    output_image = cv2.addWeighted(image_float, 1.0 - intensity, halation_image, intensity, 0)
    return output_image.astype(np.uint8)

def generate_noise(width, height, scale=10.0, intensity=0.5):
    perlin_noise = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            perlin_noise[y, x] = noise.pnoise2(x / scale, y / scale, octaves=4, persistence=0.5, lacunarity=2.0, repeatx=1024, repeaty=1024, base=42)

    perlin_noise = (perlin_noise - perlin_noise.min()) / (perlin_noise.max() - perlin_noise.min())
    gaussian_noise = np.random.normal(0, intensity, (height, width)).astype(np.float32)

    return perlin_noise * gaussian_noise

def apply_film_grain(image, intensity=0.5, grain_size=3.0):
    height, width, _ = image.shape
    noise_image = generate_noise(width, height, scale=grain_size,
