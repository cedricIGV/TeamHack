import os
import sys
#import imageDownload
import imageScrubber
#import loadData

def userChoice():
    invalid = True
    while invalid:
        choice = input(">>")
        if choice == "make":
            imageScrubber.scrub()
            invalid = False
        elif choice == "load":
            invalid = False
        elif choice == "exit":
            sys.exit()
        else:
            print("Incorrect command.")
    return

rootDir = os.getcwd()

modelDir = rootDir + "\\" + "models"

exit = False

while (exit != True):
    
    print("Available models in " + modelDir)
    
    for model in os.listdir(modelDir):
        print(model)
    print("\n")
    print("Available Commands: make, load, exit")
    userChoice()


