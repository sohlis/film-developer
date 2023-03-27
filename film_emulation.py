import os
import cv2
import json
import numpy as np
import noise
import glob

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


def get_available_luts(directory='luts'):
    return [os.path.basename(file) for file in glob.glob(f"{directory}/*.cube")]


def apply_lut(image, lut):
    image_float = image.astype(np.float32) / 255.0
    grid_size = lut.shape[0]
    indices = image_float * (grid_size - 1)

    indices_floor = np.floor(indices).astype(np.int32)
    indices_ceil = np.minimum(indices_floor + 1, grid_size - 1)

    floor_points = lut[indices_floor[..., 0], indices_floor[..., 1], indices_floor[..., 2]]
    ceil_points = lut[indices_ceil[..., 0], indices_ceil[..., 1], indices_ceil[..., 2]]

    output_image = floor_points + (indices - indices_floor) * (ceil_points - floor_points)
    return (output_image * 255).astype(np.uint8)


def apply_halation(image, threshold=200, blur_radius=31, intensity=0.5):
    image_float = image.astype(np.float32)
    ycrcb_image = cv2.cvtColor(image_float, cv2.COLOR_BGR2YCrCb)

    brightness_mask = ycrcb_image[..., 0] > threshold
    chroma_diff = np.abs(ycrcb_image[..., 1] - ycrcb_image[..., 2])
    chroma_mask = chroma_diff > 10

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
    noise_image = generate_noise(width, height, scale=grain_size, intensity=intensity / 3)
    noise_image = noise_image[:, :, np.newaxis]

    output_image = image.astype(np.float32) + (noise_image * 255 * intensity)
    return np.clip(output_image, 0, 255).astype(np.uint8)


def save_previous_selections(selections, file_path='previous_selections.json'):
    with open(file_path, 'w') as f:
        json.dump(selections, f)

def load_previous_selections():
    selections_path = "previous_selections.json"
    if os.path.exists(selections_path):
        try:
            with open(selections_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {}

def prompt_with_previous(prompt, previous_value=None):
    if previous_value is not None:
        prompt = f"{prompt} (Press Enter to use previous value: {previous_value}): "
    else:
        prompt = f"{prompt}: "
    value = input(prompt)
    return value if value else previous_value

def main():
    previous_selections = load_previous_selections()
    
    input_dir = prompt_with_previous("Input images directory", previous_selections.get("input_dir"))
    grain_level = int(prompt_with_previous("Grain level (1 to 10)", previous_selections.get("grain_level")))
    halation_level = int(prompt_with_previous("Halation level (1 to 10)", previous_selections.get("halation_level")))
    
    available_luts = get_available_luts()
    print("Available LUTs:")
    for index, lut in enumerate(available_luts, start=1):
        print(f"{index}. {lut}")
        
    lut_choice = int(prompt_with_previous("Which LUT would you like to use (Enter the number)", previous_selections.get("lut_choice")))
    lut_file = available_luts[lut_choice - 1]

    output_dir = os.path.join(input_dir, 'output')
    lut = load_lut(os.path.join('luts', lut_file))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        
        # Skip processing if the file is a directory
        if os.path.isdir(image_path):
            continue

        print(f"Processing {image_name}")
        image = cv2.imread(image_path)

        if image is not None:
            print(f"Read image from {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print("Applying halation")
            image = apply_halation(image, intensity=halation_level / 10)
            print("Applying film grain")
            image = apply_film_grain(image, intensity=grain_level / 10)
            print("Applying LUT")
            image = apply_lut(image, lut)

            output_image_path = os.path.join(output_dir, image_name)
            cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print(f"Processed and saved: {output_image_path}")
        else:
            print(f"Could not read image: {image_path}")

    save_previous_selections({
        "input_dir": input_dir,
        "grain_level": grain_level,
        "halation_level": halation_level,
        "lut_choice": lut_choice
    })
    
if __name__ == '__main__':
    main()