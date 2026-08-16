'''Shows the first three rows of the Mueller matrix for a sample. The retardance of the waveplate is determined 
by the wavelength and is used to construct the measurement matrix. The Mueller matrix is computed from the Stokes 
vectors obtained from the camera images captured at different angles of the waveplate. The resulting Mueller 
matrix elements are plotted and saved as an image on the desktop.'''
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import os, time
from ids_peak import ids_peak
from thorlabs_elliptec import ELLx
from scipy.optimize import minimize
from openpyxl import Workbook
#Necessary impots

MOTOR_PORT, MOTOR_ROTATION_DELAY = "COM3" , 0.3 #store motor port and rotation delay as global variables
correction = 83.5 #motor correction value for the waveplate, determined from calibrate_motor.py
name = "blank" #initial name
file_save_location = "Desktop" #"C:\\Users\\MMP_RobinsonGroup\\Desktop\\simple_live_qtwidgets_polarize_app\\rbc\\20260810_Yu_Zhong_COF"
#store file save location

allowed_lambdas = [310, 325, 340, 395, 500] #array of wavelengths used for measurement
retardance_lookup = {310: 360*0.2675, 325: 360*0.2635, 340: 360*0.2582, 395: 360*0.2354, 500: 360*0.25}
#dictionary of retardance values for each wavelength, used to construct the measurement matrix

def init_motor():
    #initialize the motor as an object
    motor = ELLx(x=14, serial_port=MOTOR_PORT, device_id=0)
    return motor

def motor_moving(motor, correction):
    #move motor by the correction value
    motor.move_absolute(correction)
    return

def wavelength_to_retardance():
    """Prompt the user for the acquisition wavelength and return the corresponding retardance value"""
    wavelength_input = input(f"Select acquisition wavelength {allowed_lambdas} (nm): ")
    while int(wavelength_input) not in allowed_lambdas:
        print("Wavelength choice invalid.")
        wavelength_input = input(f"Select acquisition wavelength (nm) {allowed_lambdas}: ")
    #Wait until user inputs a valid wavelength
    name_input = input("Enter a name for the acquisition (no spaces): ")
    global name
    name = name_input
    #prompt user for a name for the acquisition. Store the name as a global variable
    wavelength = int(wavelength_input)
    delta = retardance_lookup.get(wavelength, 90.0)
    #Return the corresponding retardance value for the selected wavelength, or 90.0 if not found
    return wavelength, delta

def loss_condition_number(theta_deg, delta_deg):
    "return the condition number of the measurement matrix B for the given angles and retardance value"
    B = construct_measurement_matrix(theta_deg, delta_deg)
    #construct the measurement matrix B using the given angles and retardance value
    try:
        return np.linalg.cond(B)#return the condition number for B
    except np.linalg.LinAlgError:#if the matrix is singular, the condition number will be infinite,
        return 1e10 #so return a high value to indicate a poor choice of angles

def optimize_angles(delta_deg):
    "Optimize the angles for the given retardance value to minimize the condition number"
    initial_guess = np.array([-51.7, -15.1, 15.1, 51.7])#use common values in the literature as an initial guess
    res = minimize(loss_condition_number, initial_guess, args=(delta_deg,), method='Nelder-Mead')
    #use an inbuilt minimization algorithm and the Nelder-Mead optimization method to find the optimal set of angles
    optimized_angles = (res.x + 90) % 180 - 90
    optimized_angles.sort()
    return optimized_angles
    #convert, sort, and return the new set of optimized angles

def init_camera():
    "backend code to initialize the IDS camera and datastream objects"
    ids_peak.Library.Initialize()
    dm = ids_peak.DeviceManager.Instance()
    dm.Update()
    device = dm.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
    datastream = device.DataStreams()[0].OpenDataStream()
    nodemap = device.RemoteDevice().NodeMaps()[0]

    nodemap.FindNode("ComponentSelector").SetCurrentEntry("Raw")
    nodemap.FindNode("PixelFormat").SetCurrentEntry("Mono12")
    #select the raw image component and set the pixel format to 12-bit monochrome
    payload_size = nodemap.FindNode("PayloadSize").Value()
    for _ in range(3):#allocate and announce 3 buffers for the datastream to use
        datastream.QueueBuffer(datastream.AllocAndAnnounceBuffer(payload_size))
    datastream.StartAcquisition()
    nodemap.FindNode("AcquisitionStart").Execute()
    return device, datastream

def extract_polarization_data(arr: np.ndarray):
    """Extracts the polarization data from the raw image array by splitting it
    into 2x2 blocks and returning the four polarization components."""
    rows, cols = arr.shape
    arr = arr[:rows - (rows % 2), :cols - (cols % 2)]#adjust formatting of rows and columns, losing the last row and column if they are odd
    blocks = arr.reshape(rows // 2, 2, cols // 2, 2)   #reshape the array into 2x2 blocks, where each block contains the four polarization components
    return {0: blocks[:, 1, :, 1], 45: blocks[:, 1, :, 0], 90: blocks[:, 0, :, 0], 135: blocks[:, 0, :, 1]}
    #return a dictionary containing the four polarization components, indexed by their respective angles

def compute_stokes_vectors(data):
    """Computes the stokes vectors using raw polarization data from the camera images."""
    pol_all = np.stack([[d[0], d[45], d[90], d[135]] for d in data], axis=0).astype(np.float32)
    #stack to use vectorized operations, which are faster than for loops
    p0, p45, p90, p135 = pol_all[:, 0], pol_all[:, 1], pol_all[:, 2], pol_all[:, 3]
    #store the polarization components as separate tensors
    stokes = np.stack([(p0 + p90 + p45 + p135) / 4, (p0 - p90) / 2, (p45 - p135) / 2], axis=1)
    #create a Stokes vector tensor, indexed as [measurement, stokes component, pixel row, pixel column]
    beta = (p0 + p90 - (p45 + p135)) / (p0 + p90 + p45 + p135) #compute beta values, pixel by pixel, using formula from literature
    print(f"Beta values: min = {np.min(beta):.4f}, max = {np.max(beta):.4f}, mean = {np.mean(beta):.4f}")
    #print min, max, and mean beta values for the captured image
    return stokes, beta

def capture_one_frame(datastream):
    """Captures a single frame from the IDS camera and returns it as a numpy array. 
    Ensures image quality by clearing any stale buffers before capturing the frame."""
    while True:
        try:
            stale = datastream.WaitForFinishedBuffer(20)
            datastream.QueueBuffer(stale)
            #wait for a buffer to finish, then queue it back to the datastream to clear any stale buffers
        except Exception:
            break
    buffer = datastream.WaitForFinishedBuffer(5000)
    return buffer

def capture_averaged_frame(datastream, num_frames=8): #set number of frames to capture. More increases quality, but decreases speed
    """Captures multiple frames from the IDS camera, averages them, and returns the result as a numpy array.
    By averaging multiple frames, the signal-to-noise ratio is improved, resulting in a cleaner image."""
    accum = None
    for _ in range(num_frames): #capture the specified number of frames
        buffer = capture_one_frame(datastream)#clear datastream buffer to capture a single frame at a time
        raw_address = int(buffer.BasePtr())
        buffer_from_address = (ctypes.c_char * buffer.Size()).from_address(raw_address)
        #convert the buffer to a numpy array using ctypes, which allows for direct memory access and avoids unnecessary copying of data
        img_np = np.frombuffer(buffer_from_address, dtype=np.uint16).reshape(buffer.Height(), buffer.Width()).astype(np.float32)
        #reshape the numpy array to match the dimensions of the captured image and convert to float32 for further processing
        accum = img_np.copy() if accum is None else accum + img_np
        #copy image to prevent overwriting data, and accumulate the sum of the captured frames for averaging
        datastream.QueueBuffer(buffer) #refresh datastream buffer
    return (accum / num_frames) #average the accumulated frames

def capture_stokes_sequence(motor, datastream, theta_actual, theta_relative, num_frames=8):
    """Capture the Stokes vectors and beta values for a given set of angles."""
    captured_data = []#store captured data
    num_cycles = len(theta_relative) #number of cycles to capture
    for cycle in range(1, num_cycles + 1):#cycle through each capture angle
        img_np = capture_averaged_frame(datastream, num_frames)
        captured_data.append(extract_polarization_data(img_np))
        #Capture averaged frames, extract polarization data, and store it in the captured_data list
        print(f"Captured image #{cycle} at {theta_actual[cycle-1]:.1f}°") #print which capture angle has been cycled through
        if cycle < num_cycles: #no movement is necessary after taking the last capture
            motor.move_relative(theta_relative[cycle])
            time.sleep(MOTOR_ROTATION_DELAY)#move to the next angle and wait for the motor to finish moving
    stokes_vectors, beta = compute_stokes_vectors(captured_data) #compute Stokes vectors and beta values
    return stokes_vectors, beta

def close_camera(device, datastream):
    """Stop the IDS camera acquisition and close the device and datastream objects."""
    datastream.StopAcquisition()
    ids_peak.Library.Close()

def construct_measurement_matrix(theta, delta):
    """Constructs the measurement matrix B for the given angles and retardance value."""
    B = np.zeros((4, 4))#initialieze an empty 4x4 matrix
    theta = theta*np.pi/180
    delta = delta*np.pi/180#convert from degrees to radians
    cos_2t, sin_2t = np.cos(2 * theta), np.sin(2 * theta) 
    #automatically compute cosine and sine of 2*theta for each angle
    B[:, 0] = 1.0
    B[:, 1] = cos_2t**2 + (np.cos(delta) * sin_2t**2)
    B[:, 2] = cos_2t * sin_2t * (1.0 - np.cos(delta))
    B[:, 3] = sin_2t * np.sin(delta)
    #set each column of the measurement matrix B according to the formula 
    return B

def flatten_background(img, order=2):
    """Flattens the background of an image by fitting a polynomial surface of the specified order and subtracting it from the image.
    Turn this off if it removes real details from the image. Do so by commenting out any calls to this function in the code."""

    h, w = img.shape #store height and width of the image. Should be the number of pixels in the rows and columns, respectively
    yy, xx = np.mgrid[0:h, 0:w]
    xx = (xx - w / 2) / (w / 2)
    yy = (yy - h / 2) / (h / 2)#normalize the x and y coordinates to the range [-1, 1] for numerical stability in polynomial fitting
    terms = [(xx**i) * (yy**j) for i in range(order + 1) for j in range(order + 1 - i)]
    #create a list of polynomial terms up to the specified order, where each term is a product of powers of x and y coordinates
    A = np.stack(terms, axis=-1).reshape(-1, len(terms)) #stack the polynomial terms into a 2d array
    valid = np.isfinite(img.ravel())
    #create a boolean mask to identify valid (finite) pixels in the image, which will be used for polynomial fitting
    coeffs, *_ = np.linalg.lstsq(A[valid], img.ravel()[valid], rcond=None)
    #solve the least squares problem to find the polynomial coefficients that best fit the valid pixels in the image
    background = (A @ coeffs).reshape(h, w)#create a polynomial background image using the fitted coefficients
    return img - background + np.nanmedian(img)
    #return the image with the background subtracted and the median value added back to maintain overall brightness

def plot_mueller_matrix(mueller_tensor, wavelength):
    """Code for plotting the MM elements."""

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    titles = [["m00", "m01", "m02", "m03"],
              ["m10", "m11", "m12", "m13"],
              ["m20", "m21", "m22", "m23"]]
    m00 = mueller_tensor[0, 0, :, :]
    rescaled_tensor = mueller_tensor / m00[None, None, :, :]
    #normalize the MM by m00. Mueller tensor is indexed as [row, column, pixel row, pixel column]
    for r in range(3):
        for c in range(4):#cycle through all 3 rows and 4 columns of the Mueller tensor
            if (r, c) == (0, 0): #for , do not flatten the background, as it is the reference intensity map
                continue  # leave raw intensity map untouched
            rescaled_tensor[r, c, :, :] = flatten_background(rescaled_tensor[r, c, :, :], order=2)
            #flatten background of all other elements to reduce noise
    vmin = -1.0
    vmax = 1.0#set the color scale limits for the plot
    for r in range(3):
        for c in range(4):#cycle through rows and columns of the MM
            ax = axes[r, c]
            im = ax.imshow(rescaled_tensor[r, c, :, :], cmap="bwr", vmin=vmin, vmax=vmax)
            #set the color map to blue-white-red, with the specified limits
            ax.set_title(titles[r][c], fontsize=12)
            ax.axis('off')    
    fig.subplots_adjust(right=0.85, hspace=0.3, wspace=0.3)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.suptitle(f"Mueller Matrix Elements (Top 3 Rows) at {wavelength} nm", fontsize=16, y=0.95)
    output_path = os.path.join(os.path.expanduser("~"), file_save_location, f"{name}_MM_{wavelength}nm.png")
    #save plot to the desktop with the specified name and wavelength
    plt.savefig(output_path, bbox_inches='tight', dpi=300)

def plot_stokes_vectors(stokes, theta_actual, wavelength):
    """Code for plotting the Stokes vectors."""

    stokes = stokes/stokes[:, 0:1, :, :] #normalize each Stokes vector by its S0 componennt
    num_cycles = stokes.shape[0] #set the number of cycles to capture as the number of angles in the set
    titles = ["S0", "S1", "S2"]
    fig, axes = plt.subplots(3, num_cycles, figsize=(4 * num_cycles, 12))
    for cycle in range(num_cycles):
        for s in range(3): #cycle through the 3 stokes components
            ax = axes[s, cycle]
            im = ax.imshow(stokes[cycle, s, :,:], cmap="bwr", vmin = -1.0, vmax = 1.0)
            #utilize the blue-white-red color map, with limits of -1 to 1, to visualize the Stokes components
            ax.set_title(f"{theta_actual[cycle]:.2f}° — {titles[s]}")
            ax.set_xlabel("Block Column")
            ax.set_ylabel("Block Row")
            fig.colorbar(im, ax=ax)
    plt.tight_layout()
    output_path = os.path.join(os.path.expanduser("~"), file_save_location, f"{name}_Stokes_{wavelength}nm.png")
    #save plot to the desktop with the specified name and wavelength
    plt.savefig(output_path, bbox_inches='tight', dpi=300)

def export_csv(mueller_tensor, wavelength):
    """Exports the Mueller tensor to an Excel file, with each element of the tensor saved in a separate sheet."""

    m00 = mueller_tensor[0, 0, :, :]
    #commenting out the normalization step to keep the raw Mueller matrix values for export
    #mueller_tensor = mueller_tensor / m00[None, None, :, :]
    wb = Workbook()
    wb.active
    for r in range(3):
        for c in range(4):#cycle through all 3 rows and 4 columns of the Mueller tensor
            sheet_name = f"m{r}{c}"
            ws = wb.create_sheet(title=sheet_name)#create a sheet for the given MM componenet
            grid = mueller_tensor[r, c, :, :]#each Excel cell will contain the value of the corresponding pixel in the MM component
            for row in grid:
                ws.append(row.tolist())
    output_path = os.path.join(os.path.expanduser("~"), file_save_location, f"{name}_{wavelength}nm.xlsx")
    #save the Excel file to the desktop with the specified name and wavelength
    wb.save(output_path)

def main():
    #to change between multiple stokes capture and MM capture, change which theta_actual is used, and un/comment all MM related calls
    motor = init_motor()#initialize motor
    motor_moving(motor, correction)#move to the correction position
    wavelength, delta = wavelength_to_retardance()#collect wavelength and retardance values
    theta_actual = optimize_angles(delta)#use the optimized angles for the given retardance
    #theta_actual = np.array([-90, -45, 0, 45, 90])

    thetas = len(theta_actual)
    theta_relative = np.zeros(thetas)
    theta_relative[0] = theta_actual[0]
    for i in range(1, thetas):
        theta_relative[i] = (theta_actual[i] - theta_actual[i-1])
    #Adjust for relative movement

    motor.move_relative(theta_relative[0])#move to the first capture angle in the set
    device, datastream = init_camera()#initialize camera
    stokes, beta = capture_stokes_sequence(motor, datastream, theta_actual, theta_relative)
    #capture the Stokes vectors and beta values for the given set of angles
    close_camera(device, datastream)#close the camera

    B = construct_measurement_matrix(theta_actual, delta)
    B_inv = np.linalg.inv(B)
    mueller_tensor = np.einsum('ck,krhw->rchw',B_inv, stokes)
    #construct and invert the measurement matrix, using that to calculate the MM

    plot_mueller_matrix(mueller_tensor, wavelength)
    plot_stokes_vectors(stokes, theta_actual, wavelength)
    export_csv(mueller_tensor, wavelength)
    #plot MM and stokes vectors, export MM to an Excel workbook

if __name__ == "__main__":
    main()