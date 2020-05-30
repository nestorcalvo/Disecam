# Libraries
# Library to scan files in a directory
from os import scandir
# Library to work with images
import cv2


def ls2(path):
    # Function to save the name of all files
    return [obj.name for obj in scandir(path) if obj.is_file()]


# Open a file
source_path = 'C:/Users/USUARIO/Documents/PDI_FINAL/BD/melanoma/train/'
destination_path = 'C:/Users/USUARIO/Documents/PDI_FINAL/BDN/melanoma/train/'

files = ls2(source_path)
# For each file in the path
for file in files:
    # Read the image
    imagen = cv2.imread(source_path + file)
    # Convert to grayscale
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # Resize the image to a standard value
    new_image = cv2.resize(gray, (600, 450))
    # Save the image after all the changes
    cv2.imwrite(destination_path + file, new_image)

