import numpy as np
from PIL import Image
import os
import csv
from pprint import pprint 


def file_indexer():

    files = []
    for dirname, dirnames, filenames in os.walk('.'):
        for subdirname in dirnames:
            files.append(os.path.join(dirname, subdirname))

        for filename in filenames:
            files.append(os.path.join(dirname, filename))
    pprint(files)

    writeToFile = []
    writeToTestFile =[]    

    for file in files:
        if ".png" in file and "train" in file:
            writeToFile.append(file)
        if ".png" in file and "test" in file:
            writeToTestFile.append(file)

    with open("Files.csv", "w") as f:
        write = csv.writer(f)
        write.writerows(writeToFile)

    with open("TestFiles.csv", "w") as f:
        write = csv.writer(f)
        write.writerows(writeToTestFile)

def write_to_csv(trainOrTest, label, contents):
    
    arrayToDump = np.insert(contents, 0, label)
    arrayToDump = arrayToDump.reshape(1,785)

    dataFileName = ""
    if trainOrTest:
        dataFileName = "train.csv"
    else:
        dataFileName = "test.csv"
    
    with open(dataFileName, "a") as f:
        np.savetxt(f, arrayToDump, fmt="%d", delimiter=",")


def open_image(pathToFileIndex,trainOrTest):
    
    with open(pathToFileIndex, "r") as f:
        fileList = csv.reader(f)

        for file in fileList:

            fileName = "".join(file)
            label = fileName[(fileName.rfind("/")-1)]
            image = Image.open(fileName)

            numpydata = np.asarray(image)
            write_to_csv(trainOrTest,label, numpydata)


file_indexer()
open_image("Files.csv", True)
open_image("TestFiles.csv", False)
