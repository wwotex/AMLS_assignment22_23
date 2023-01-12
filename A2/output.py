import os
import globals 

globals.initialize()

global full_output
full_output = ''

def printWW(str):
    global full_output
    print(str)
    full_output += str


def saveOutputToFile():
    with open(os.path.join(globals.saved_data_dir, 'model_selection_output.txt'), 'a') as f:
        f.write(full_output)