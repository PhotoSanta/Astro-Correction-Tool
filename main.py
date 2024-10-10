import numpy as np
import torch
import torch.nn.functional as F
from skimage import io, img_as_float, img_as_ubyte
from skimage.restoration import denoise_tv_chambolle
import matplotlib.pyplot as plt
from tkinter import Tk, Button, Scale, HORIZONTAL, Label, filedialog, messagebox, Checkbutton, IntVar
from PIL import Image, ImageTk
import threading
import os

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize global variables
image = None
image_path = None
processed_image = None

# Define an elliptical Gaussian PSF to model astigmatism
def elliptical_gaussian_psf(size, sigma_x, sigma_y, device):
    """Create an elliptical Gaussian kernel."""
    x = torch.linspace(-size // 2, size // 2, steps=size, device=device)
    y = torch.linspace(-size // 2, size // 2, steps=size, device=device)
    x, y = torch.meshgrid(x, y, indexing='ij')
    psf = torch.exp(-((x ** 2) / (2 * sigma_x ** 2) + (y **2) / (2 * sigma_y ** 2)))
    psf /= psf.sum()
    return psf

# Perform Richardson-Lucy deconvolution
def richardson_lucy(image_tensor, psf_tensor, num_iter):
    image_tensor = torch.clamp(image_tensor, min=1e-7)
    psf_mirror = torch.flip(psf_tensor, dims=[-2, -1])
    estimate = torch.full_like(image_tensor, 0.5)
    for _ in range(num_iter):
        relative_blur = F.conv2d(estimate, psf_tensor, padding='same')
        relative_blur = torch.clamp(relative_blur, min=1e-7)
        correction = F.conv2d(image_tensor / relative_blur, psf_mirror, padding='same')
        estimate *= correction
    return estimate

# Function to process a single channel
def process_channel(channel, psf_tensor, num_iterations):
    # Convert to tensor and move to GPU
    channel_tensor = torch.tensor(channel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    # Perform deconvolution
    deconvolved_tensor = richardson_lucy(channel_tensor, psf_tensor, num_iterations)
    # Move result back to CPU and convert to NumPy array
    deconvolved_channel = deconvolved_tensor.squeeze().cpu().numpy()
    return deconvolved_channel

# Function to load an image
def load_image():
    global image, image_path
    filepath = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.tif;*.tiff"), ("All files", "*.*")]
    )
    if filepath:
        image = img_as_float(io.imread(filepath))
        image_path = filepath
        display_image(image, original=True)

# Function to display an image
def display_image(img_array, original=False):
    img = img_as_ubyte(np.clip(img_array, 0, 1))
    img = Image.fromarray(img)
    (img_width, img_height) = img.size
    img_ratio = img_width / img_height
    print(img_ratio)
    print(int(img_height * img_ratio))
    img = img.resize((525, int(525 / img_ratio)))
    img_tk = ImageTk.PhotoImage(img)
    if original:
        original_image_label.config(image=img_tk)
        original_image_label.image = img_tk
    else:
        processed_image_label.config(image=img_tk)
        processed_image_label.image = img_tk

# Function to save the processed image
def save_processed_image():
    global processed_image, image_path
    if processed_image is None:
        messagebox.showwarning("No Processed Image", "Please process an image first.")
        return

    # Extract directory and original filename
    dir_name, original_filename = os.path.split(image_path)
    name, ext = os.path.splitext(original_filename)

    # Create new filename
    new_name = f"{name}_edit{ext}"
    new_path = os.path.join(dir_name, new_name)

    # Check if file exists and append a number if necessary
    counter = 1
    while os.path.exists(new_path):
        new_name = f"{name}_edit{counter}{ext}"
        new_path = os.path.join(dir_name, new_name)
        counter += 1

    # Save the processed image
    denoised_uint8 = img_as_ubyte(np.clip(processed_image, 0, 1))
    io.imsave(new_path, denoised_uint8)

    # Inform the user
    messagebox.showinfo("Image Saved", f"Processed image saved as:\n{new_path}")

# Function to process the image
def process_image():
    global processed_image

    if image is None:
        messagebox.showwarning("No Image Loaded", "Please load an image first.")
        return

    # Get parameters from sliders
    sigma_x = sigma_x_slider.get()
    sigma_y = sigma_y_slider.get()
    psf_size = psf_size_slider.get()
    num_iterations = iterations_slider.get()

    # Get color filter settings
    red_filter_enabled = red_filter_var.get()
    red_filter_value = red_filter_slider.get() / 100  # Convert percentage to fraction

    green_filter_enabled = green_filter_var.get()
    green_filter_value = green_filter_slider.get() / 100

    blue_filter_enabled = blue_filter_var.get()
    blue_filter_value = blue_filter_slider.get() / 100

    # Inform the user that processing has started
    process_button.config(state='disabled', text='Processing...')
    root.update()

    # Run processing in a separate thread to keep the GUI responsive
    def processing_thread():
        global processed_image
        try:
            # Create PSF tensor
            psf_tensor = elliptical_gaussian_psf(psf_size, sigma_x, sigma_y, device)
            psf_tensor = psf_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)

            # Check if the image is color
            if image.ndim == 3:
                # Split the image into R, G, B channels
                r_channel = image[:, :, 0]
                g_channel = image[:, :, 1]
                b_channel = image[:, :, 2]

                # Process each channel
                deconvolved_r = process_channel(r_channel, psf_tensor, num_iterations)
                deconvolved_g = process_channel(g_channel, psf_tensor, num_iterations)
                deconvolved_b = process_channel(b_channel, psf_tensor, num_iterations)

                # Combine the channels back into an image
                deconvolved_image = np.stack((deconvolved_r, deconvolved_g, deconvolved_b), axis=2)
            else:
                # Process single-channel image
                deconvolved_channel = process_channel(image, psf_tensor, num_iterations)
                deconvolved_image = deconvolved_channel

            # Apply Total Variation Denoising
            denoised_image = denoise_tv_chambolle(
                deconvolved_image, weight=0.1, channel_axis=-1 if deconvolved_image.ndim == 3 else None
            )

            # Apply color filtering if enabled
            if denoised_image.ndim == 3:
                if red_filter_enabled:
                    # Reduce red channel
                    denoised_image[:, :, 0] *= (1 - red_filter_value)
                if green_filter_enabled:
                    # Reduce green channel
                    denoised_image[:, :, 1] *= (1 - green_filter_value)
                if blue_filter_enabled:
                    # Reduce blue channel
                    denoised_image[:, :, 2] *= (1 - blue_filter_value)
            else:
                # If image is grayscale, do nothing or apply a grayscale equivalent if desired
                pass

            # Update the processed image
            processed_image = denoised_image

            # Display the processed image
            display_image(denoised_image, original=False)

            # Save the processed image
            save_processed_image()

        except Exception as e:
            messagebox.showerror("Processing Error", str(e))
        finally:
            # Reset the process button
            process_button.config(state='normal', text='Process Image')
            root.update()

    threading.Thread(target=processing_thread).start()

# Functions to enable/disable the color filter sliders based on checkbox states
def toggle_red_filter():
    if red_filter_var.get():
        red_filter_slider.config(state='normal')
    else:
        red_filter_slider.config(state='disabled')

def toggle_green_filter():
    if green_filter_var.get():
        green_filter_slider.config(state='normal')
    else:
        green_filter_slider.config(state='disabled')

def toggle_blue_filter():
    if blue_filter_var.get():
        blue_filter_slider.config(state='normal')
    else:
        blue_filter_slider.config(state='disabled')

# Create the main window
root = Tk()
root.title("Stars Correction Tool")

# Create and place the load image button
load_button = Button(root, text="Load Image", command=load_image)
load_button.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

# Create sliders for parameters
sigma_x_label = Label(root, text="Sigma X:")
sigma_x_label.grid(row=1, column=0, padx=10, sticky='w')
sigma_x_slider = Scale(root, from_=1, to=10, orient=HORIZONTAL)
sigma_x_slider.set(2)
sigma_x_slider.grid(row=1, column=1, padx=10, pady=5, sticky='ew')

sigma_y_label = Label(root, text="Sigma Y:")
sigma_y_label.grid(row=2, column=0, padx=10, sticky='w')
sigma_y_slider = Scale(root, from_=1, to=10, orient=HORIZONTAL)
sigma_y_slider.set(5)
sigma_y_slider.grid(row=2, column=1, padx=10, pady=5, sticky='ew')

psf_size_label = Label(root, text="PSF Size:")
psf_size_label.grid(row=3, column=0, padx=10, sticky='w')
psf_size_slider = Scale(root, from_=3, to=61, orient=HORIZONTAL, resolution=2)
psf_size_slider.set(31)
psf_size_slider.grid(row=3, column=1, padx=10, pady=5, sticky='ew')

iterations_label = Label(root, text="Iterations:")
iterations_label.grid(row=4, column=0, padx=10, sticky='w')
iterations_slider = Scale(root, from_=5, to=50, orient=HORIZONTAL)
iterations_slider.set(30)
iterations_slider.grid(row=4, column=1, padx=10, pady=5, sticky='ew')

# Add red filter checkbox and slider
red_filter_var = IntVar()
red_filter_checkbox = Checkbutton(root, text="Enable Red Filter", variable=red_filter_var, command=toggle_red_filter)
red_filter_checkbox.grid(row=5, column=0, padx=10, sticky='w')

red_filter_label = Label(root, text="Red Filter Strength (%):")
red_filter_label.grid(row=6, column=0, padx=10, sticky='w')
red_filter_slider = Scale(root, from_=0, to=100, orient=HORIZONTAL, state='disabled')
red_filter_slider.set(0)
red_filter_slider.grid(row=6, column=1, padx=10, pady=5, sticky='ew')

# Add green filter checkbox and slider
green_filter_var = IntVar()
green_filter_checkbox = Checkbutton(root, text="Enable Green Filter", variable=green_filter_var, command=toggle_green_filter)
green_filter_checkbox.grid(row=7, column=0, padx=10, sticky='w')

green_filter_label = Label(root, text="Green Filter Strength (%):")
green_filter_label.grid(row=8, column=0, padx=10, sticky='w')
green_filter_slider = Scale(root, from_=0, to=100, orient=HORIZONTAL, state='disabled')
green_filter_slider.set(0)
green_filter_slider.grid(row=8, column=1, padx=10, pady=5, sticky='ew')

# Add blue filter checkbox and slider
blue_filter_var = IntVar()
blue_filter_checkbox = Checkbutton(root, text="Enable Blue Filter", variable=blue_filter_var, command=toggle_blue_filter)
blue_filter_checkbox.grid(row=9, column=0, padx=10, sticky='w')

blue_filter_label = Label(root, text="Blue Filter Strength (%):")
blue_filter_label.grid(row=10, column=0, padx=10, sticky='w')
blue_filter_slider = Scale(root, from_=0, to=100, orient=HORIZONTAL, state='disabled')
blue_filter_slider.set(0)
blue_filter_slider.grid(row=10, column=1, padx=10, pady=5, sticky='ew')

# Create and place the process image button
process_button = Button(root, text="Process Image", command=process_image)
process_button.grid(row=11, column=0, columnspan=2, padx=10, pady=10, sticky='ew')

# Create labels to display images
original_image_label = Label(root)
original_image_label.grid(row=12, column=0, padx=10, pady=10)
processed_image_label = Label(root)
processed_image_label.grid(row=12, column=1, padx=10, pady=10)

# Start the GUI event loop
root.mainloop()
