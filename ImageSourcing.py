from PIL import Image
import os

path = '.'

for dir in sorted(os.listdir(path)):
        try:
                path2 = os.listdir(dir)
                print("number of files in",dir,"is",len(path2))
        except:
                pass