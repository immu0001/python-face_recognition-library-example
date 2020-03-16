import face_recognition
import os
import cv2

KNOWN_FACES_DIR ="known-face"
UNKNOWN_FACES_DIR = "unknown-face"
TOLERENCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "cnn" # try using "hog"

print("loading known faces")

known_faces  = []
known_names = ["Justin Beiber"]


for name in os.listdir(KNOWN_FACES_DIR):
	for filename in os.listdir(f"{KNOWN_FACES_DIR}"):
		image  =face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{filename}")
		encoding = face_recognition.face_encodings(image)[0]
		known_faces.append(encoding)
		#known_names.append(name)



print("Processing unknown faces")
for filename in os.listdir(UNKNOWN_FACES_DIR):
	print(filename)
	image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
	locations = face_recognition.face_locations(image, model=MODEL)
	encoding = face_recognition.face_encodings(image, locations)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	for face_encoding, face_locations in zip(encoding, locations):
		results = face_recognition.compare_faces(known_faces, face_encoding, TOLERENCE)
		match =  None
		if True in results:
			match = known_names[results.index(True)]
			print(f"Match found : {match}")

			top_left = (face_locations[3], face_locations[0])
			bottom_right = (face_locations[1], face_locations[2])

			color = [0, 255, 0]

			cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

			top_left = (face_locations[3], face_locations[2])
			bottom_right = (face_locations[1], face_locations[2]+22)
			cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
			cv2.putText(image, match,  (face_locations[3]+10, face_locations[2]+15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200,200,200), FONT_THICKNESS)
	cv2.imshow(filename, image)
	cv2.waitKey(0)
	cv2.destroyWindow()



