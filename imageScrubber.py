import os
from PIL import Image

def scrub():

    rootdir = os.getcwd()

    folderName = input("Select folder to scrub: ")

    currentDir = rootdir + '\\' + folderName


    for file in os.listdir(currentDir):
        currentFile = currentDir + "\\" + file
        try:
            currentImage = Image.open(currentFile)
        except IOError:
            os.remove(currentFile)
            print("Deleted " + currentFile)
    return
