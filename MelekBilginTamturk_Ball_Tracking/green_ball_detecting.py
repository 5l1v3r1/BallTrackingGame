from collections import deque
import numpy as np
import argparse
import imutils
import cv2

def detect_the_most_green_thing(green_things):
	the_most_green_thing = max(green_things, key=cv2.contourArea)
	# finding minimum enclosed circle
	((x, y), radius) = cv2.minEnclosingCircle(the_most_green_thing)
	# centroid
	for_centroid = cv2.moments(the_most_green_thing)
	center = (int(for_centroid["m10"] / for_centroid["m00"]), int(for_centroid["m01"] / for_centroid["m00"]))
	return (x,y),radius,center
def draw_perimeter_of_green(frame,(x,y),radius,center):
	# if radius is long enough
	if radius > 15:
		cv2.circle(frame, (int(x), int(y)), int(radius),(51, 205, 51), 2)
		cv2.circle(frame, center, 5, (0, 0, 255), -1)

def draw_line(frame,tracked_points):
	i = 1
	length = len(tracked_points)
	while (i < length):
		thickness_of_the_line = int(np.sqrt(args["buffer"] / float(i + 1)) * 3.5)
		cv2.line(frame, tracked_points[i - 1], tracked_points[i], (153, 255, 204), thickness_of_the_line)
		i = i + 1

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=32,help="max buffer size")
args = vars(ap.parse_args())

# defining the boundries of green color in HSV
color_lower_boundry  = (29, 86, 6)
color_upper_boundry  = (64, 255, 255)
# initializing the list of tracked points
tracked_points = deque(maxlen=args["buffer"])
# Opening the webcam
camera = cv2.VideoCapture(0)

flag = True
while flag:
	(need,frame) = camera.read()

	# resize the frame, blur it, and convert it to the HSV
	frame = imutils.resize(frame, width=1000)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	#look for a green color
	filter_for_green = cv2.inRange(hsv, color_lower_boundry , color_upper_boundry )

	# erode and dilate for removing blobs
	erodeded = cv2.erode(filter_for_green, None, iterations=2)
	dilated = cv2.dilate(erodeded, None, iterations=2)

	# find green things
	green_things = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None

	# if one or more has found
	if len(green_things) > 0:
		(x,y),radius,center = detect_the_most_green_thing(green_things)
		draw_perimeter_of_green(frame,(x,y),radius,center)
	tracked_points.appendleft(center)
	# draw the line according to the tracked points
	draw_line(frame,tracked_points)
	cv2.imshow("BALL TRACKING", frame)

	# if the 'esc' key is pressed, stop the loop
	if cv2.waitKey(1) == 27:
		flag = False

camera.release()
cv2.destroyAllWindows()