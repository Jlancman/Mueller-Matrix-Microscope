'''Program to run through a set of angles and determine which gives 
the best calibration for the waveplate. The program will move the 
motor to a set of angles, capture Stokes vectors at each angle, 
and compare to the known stokes vectores for those angles. 
The calibration offset which yields the closest result is determined
to be the accurate offset angle.'''

import numpy as np
import matplotlib.pyplot as plt
import os
from runme import (
    motor_moving, init_motor, init_camera, close_camera, wavelength_to_retardance,
    capture_stokes_sequence
)
#Necessary imports. Imports needed modules from runme.py

CAL_START = 80
CAL_STOP = 90
CAL_STEP = 0.1
#Set start, stop and step values for the calibration sweep.

def evaluate_identity_error(stokes):
    "Normalize the measured Stokes vectors and compare to the ideal values for the given angles."
    stokes = stokes/stokes[:, 0:1, :, :]
    #Stokes variable tensor is normalized by the first element of the Stokes vector (S0) 
    #to ensure that the intensity is consistent across all measurements.
    S1 = stokes[:, 1, :, :]
    S2 = stokes[:, 2, :, :]
    #Stokes tensor is indexed as [measurement, stokes_component, pixel row, pixel column]
    target_S1 = np.array([1, 0, 1, 0, 1]).reshape(5, 1, 1)
    target_S2 = np.zeros((5, 1, 1))
    #ideal values for the stokes components of the angle set given below
    mse = np.mean((S1 - target_S1) ** 2) + np.mean((S2 - target_S2) ** 2)
    #Indexing score for how close the measured stokes vectors are to the ideal values.
    return float(1.0 / (1.0 + mse))
    #returns a score between 0 and 1, with 1 being a perfect match to the ideal stokes vectors.

def main():
    "Run procedure to collect Stokes vectors over a range of calibration angles, using interface with the runme program"
    wavelength, delta = wavelength_to_retardance()#set wavelength and retardance
    theta_actual = np.array([-90, -45, 0, 45, 90])#set angles for Stokes vector capture
    thetas = len(theta_actual)#how many angles are in the set
    theta_relative = np.zeros(thetas)#empty initalizer, with length of angles in the set
    theta_relative[0] = theta_actual[0]
    for i in range(1, thetas):
        theta_relative[i] = (theta_actual[i] - theta_actual[i-1])
    #this step uses relative movement, such that the motor moves to the correct angles
    #relative to where it was before. This is necessary for accurate movement and measurement
    n_steps = int(round((CAL_STOP - CAL_START) / CAL_STEP)) + 1 #number of steps in calibration sweep
    calibration_angles = np.round(CAL_START + np.arange(n_steps) * CAL_STEP, 1) #set of calibration angles
    motor = init_motor() #initialize the motor
    device, datastream = init_camera() #initialize the camera
    best_angle, best_score = None, 0 #set initial best angle as none, best score as 0. These will be updated throughout the procedure
    results = [] #store results for plotting
    for angle in calibration_angles: #cycle through each angle in the calibration sweep
        motor_moving(motor, angle) #move the motor to the current calibration angle in the sweep
        motor.move_relative(theta_relative[0]) #move to the first angle in the set of capture angles
        stokes, beta = capture_stokes_sequence(motor, datastream, theta_actual, theta_relative) #capture stokes vectors and beta values
        score = evaluate_identity_error(stokes) #evaluate how close the measured stokes vectors are to the ideal values
        print(f"calibration = {angle:5.1f}°  ->  {score:.6f}") #print this value
        results.append(score)#add the score to the results list for plotting
        if score > best_score:
            best_score = score
            best_angle = float(angle)
        #best in set algorithm. If the current score is better than the best score, update the best score and best angle.
    close_camera(device, datastream)#close camera after the calibration sweep is complete
    print("\n=== Calibration sweep complete ===")
    if best_angle is not None:
        print(f"Best calibration angle: {best_angle:.1f}°  (score = {best_score:.6f})")
    #print the results of the best angle found
    else:
        print("No valid results were obtained.")
    #check in case there are no valid results

    plt.figure(figsize=(7, 4))
    plt.scatter(calibration_angles, results)
    plt.axvline(best_angle, color='r', linestyle='--', label=f'best = {best_angle:.1f}°')
    plt.xlabel('Calibration angle (°)')
    plt.ylabel('Identity error score')
    plt.title('Analyzer calibration sweep')
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(os.path.expanduser("~"), "Desktop", f"Calibration.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    #code for plotting the results of the calibration sweep and saving the figure to the desktop

if __name__ == "__main__":
    main()