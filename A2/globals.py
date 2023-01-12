import os

def initialize(): 
    global full_output, base_dir, saved_data_dir, image_dir
    base_dir = os.path.dirname(__file__)
    saved_data_dir = os.path.join(base_dir, 'saved_data')
    image_dir = os.path.join(base_dir, 'graphs')
    full_output = ''