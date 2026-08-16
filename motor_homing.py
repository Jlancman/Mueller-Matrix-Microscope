"""This script is used to control a Thorlabs Elliptec motor.
It imports the necessary libraries, sets the motor port, initializes the motor, homes it,
and then moves it to an absolute position of 83.1 units. The script 
includes sleep intervals to allow time for the motor to complete its movements."""

#Necessary imports for motor control and timing
from ids_peak import ids_peak
from thorlabs_elliptec import ELLx
import time

MOTOR_PORT = "COM3" #set the serial port for the motor connection. Change for the appropriate port on your system.
motor = ELLx(x=14, serial_port=MOTOR_PORT, device_id=0) #Initializes the motor
motor.home(0)#Places the motor in its home position
time.sleep(1)#Delay for 1 sec
motor.move_absolute(83.1)#Move to a given absolute position, relative to home 
time.sleep(1)