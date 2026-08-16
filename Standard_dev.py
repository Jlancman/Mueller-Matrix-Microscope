
'''Given a set of repeated Mueller matrix captures (using the same acquisition
procedure as runme_mueller.py), computes the pixel-by-pixel standard deviation
across those captures and plots it as a 3x4 grid — same layout and color
scheme as plot_mueller_matrix — to visualize measurement repeatability/noise
per Mueller matrix element.'''
import numpy as np
import matplotlib.pyplot as plt
import os
from runme import (
    motor_moving, init_motor, init_camera, close_camera, wavelength_to_retardance,
    optimize_angles, capture_stokes_sequence, construct_measurement_matrix,
    flatten_background, correction, name
)
#Necessary imports. Imports needed modules from runme.py

def capture_single_mueller(motor, datastream, theta_actual, theta_relative, delta):
    "Captures a single Mueller matrix using the same procedure as runme.py, returning the Mueller tensor."
    stokes, beta = capture_stokes_sequence(motor, datastream, theta_actual, theta_relative)
    #capture Stokes vectors and beta values
    B = construct_measurement_matrix(theta_actual, delta) #Form measurement matrix
    B_inv = np.linalg.inv(B) #invert measurement matrix
    mueller_tensor = np.einsum('ck,krhw->rchw', B_inv, stokes) #calculate Mueller tensor using Einstein summation convention
    return mueller_tensor

def normalize_and_flatten(mueller_tensor):
    "Normalizes the Mueller tensor by its m00 element and flattens the background of each element (except m00) to reduce noise."
    m00 = mueller_tensor[0, 0, :, :] #Mueller tensor is indexed as [row, column, pixel row, pixel column]
    rescaled = mueller_tensor / m00[None, None, :, :] #Normalize the Mueller tensor by its m00 element
    for r in range(3):
        for c in range(4):#cycle through all 3 rows and 4 columns of the Mueller tensor
            if (r, c) == (0, 0):#for m00, do not flatten the background, as it is the reference intensity map
                continue  # leave raw intensity map untouched, as in plot_mueller_matrix
            rescaled[r, c, :, :] = flatten_background(rescaled[r, c, :, :], order=2)
    #flattens the background of each element (except m00) to reduce noise, using a polynomial fit of order 2
    #Any higher order may eliminate real features
    return rescaled

def plot_std_matrix(std_tensor, wavelength, num_captures):
    """plot the standard deviation of the Mueller matrix elements as a 3x4 grid,
    with the same layout and color scheme as plot_mueller_matrix."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    titles = [["m00", "m01", "m02", "m03"],
              ["m10", "m11", "m12", "m13"],
              ["m20", "m21", "m22", "m23"]]
    vmax = float(np.nanpercentile(std_tensor, 99))
    for r in range(3):
        for c in range(4):
            ax = axes[r, c]
            im = ax.imshow(std_tensor[r, c, :, :], cmap="gray", vmin=0.0, vmax=vmax)
            ax.set_title(titles[r][c], fontsize=12)
            ax.axis('off')
    fig.subplots_adjust(right=0.85, hspace=0.3, wspace=0.3)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.suptitle(f"Mueller Matrix Std. Dev. (n={num_captures}) at {wavelength} nm", fontsize=16, y=0.95)
    output_path = os.path.join(os.path.expanduser("~"), "Desktop", f"{name}_MM_std_{wavelength}nm.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    #code for plotting the matrix

def get_num_captures():
    """Prompt the user for the number of repeated Mueller matrix captures to perform. Must be aninteger >= 2."""
    n_input = input("Number of repeated Mueller matrix captures for std dev (>=2): ")
    while not n_input.isdigit() or int(n_input) < 2:
        print("Enter an integer >= 2.")
        n_input = input("Number of repeated Mueller matrix captures: ")
    #wait until user inputs a valid number of captures. The higher, the more accurate the data
    return int(n_input)

def main():
    motor = init_motor()#initialize motor
    motor_moving(motor, correction)#move motor to the correction position
    wavelength, delta = wavelength_to_retardance()#collect wavelength and retardance values
    theta_actual = optimize_angles(delta)#use the optimized angles for the given retardance value

    thetas = len(theta_actual)
    theta_relative = np.zeros(thetas)
    theta_relative[0] = theta_actual[0]
    for i in range(1, thetas):
        theta_relative[i] = (theta_actual[i] - theta_actual[i - 1])
    #Adjust for relative movement

    num_captures = get_num_captures()#store number of captures to perform, as input by the user
    device, datastream = init_camera()#initialize camera
    processed_tensors = []#store processed Mueller tensors for each capture
    for cap in range(1, num_captures + 1):#move through the number of captures
        motor.move_relative(theta_relative[0])#move to the first angle in the set of capture angles
        mueller_tensor = capture_single_mueller(motor, datastream, theta_actual, theta_relative, delta)
        #capture a single Mueller matrix using the same procedure as runme.py, returning the Mueller tensor
        processed_tensors.append(normalize_and_flatten(mueller_tensor))
        #Store the processed Mueller tensor for this capture, after normalizing and flattening the background
        print(f"Completed capture {cap}/{num_captures}")
    close_camera(device, datastream)#close camera
    stacked = np.stack(processed_tensors, axis=0)#stack together all tensors for standard deviation calculation
    std_tensor = np.std(stacked, axis=0, ddof=1)#calculate standard dev, pixel by pixel for all Mueller matrix elements
    plot_std_matrix(std_tensor, wavelength, num_captures)
    #plot the standard dev as a 3x4 grid, with the same layout and color scheme as plot_mueller_matrix

if __name__ == "__main__":
    main()