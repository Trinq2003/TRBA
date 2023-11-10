"""
    Label text file processing (take the paragraph below CSR line)
"""

import os
from PIL import Image
import pandas as pd
import csv
import random 
import pickle

dataset_path = 'test_label'
processed_label_path = 'processed_label'
os.makedirs(processed_label_path, exist_ok=True)
label_folder_path = os.path.join(dataset_path, 'label')

for root, dirs, files in os.walk(label_folder_path):
    for file in files:
        if file.lower().endswith('.txt'):
            input_path = os.path.join(root, file)
            output_folder_path = os.path.join(processed_label_path, os.path.relpath(root, label_folder_path))
            output_path = os.path.join(output_folder_path, file)

            os.makedirs(output_folder_path, exist_ok=True)

            with open(input_path, 'r') as input_file:
                lines = input_file.readlines()

            csr_lines = []
            is_csr_part = False
            for line in lines:
                if line.startswith('CSR:'):
                    is_csr_part = True
                elif line.startswith('OCR:'):
                    is_csr_part = False
                if is_csr_part:
                    csr_lines.append(line)

            with open(output_path, 'w') as output_file:
                output_file.writelines(csr_lines[2:])

print("Label processing completed!")

# Convert from .tif file -> .jpg file

import os
from PIL import Image

dataset_path = ''
processed_data_path = 'processed_data'
os.makedirs(processed_data_path, exist_ok=True)
data_folder_path = os.path.join(dataset_path, 'data')

for root, dirs, files in os.walk(data_folder_path):
    for file in files:
        if file.lower().endswith('.tif'):
            input_path = os.path.join(root, file)
            output_folder_path = os.path.join(processed_data_path, os.path.relpath(root, data_folder_path))
            output_path = os.path.join(output_folder_path, file[:-4] + '.jpg')
            os.makedirs(output_folder_path, exist_ok=True)

            with Image.open(input_path) as image:
                image = image.convert('RGB')
                image.save(output_path, 'JPEG')
            # os.remove(input_path)

print("Data conversion completed!")


# Assign each image with its respecting label
processed_data_path = '../resources/train/data/'
processed_label_path = 'test_label_p'

output_file_path = '../resources/train/label.csv'

with open(output_file_path, 'w', newline='') as output_file:
    count = -1
    writer = csv.writer(output_file, delimiter='\t')
    writer.writerow(['No','Image', 'Label'])
    
    for root, dirs, files in os.walk(processed_data_path):
        for file in files:
            count += 1
            label_file_path = os.path.join(processed_label_path, file[:-7] + '.txt')
            if file.lower().endswith('.jpg'):
                try:
                    image_file_path = os.path.join(root, file)
                    with open(label_file_path, 'r') as label_file:
                        lines = label_file.readlines()
                    lines = [line.strip() for line in lines]
                    writer.writerow([count, file, lines[int(file.lower()[-6:-4])-1]])
                except:
                    print(file)

print("Output file created!")

# Move all the image in the hierachical folder to one folder

import os
import shutil

processed_data_path = 'train_label'
train_data_path = 'test_label_p'
os.makedirs(train_data_path, exist_ok=True)

for root, dirs, files in os.walk(processed_data_path):
    for file in files:
        if file.lower().endswith('.txt'):
            input_path = os.path.join(root, file)
            output_path = os.path.join(train_data_path, file)
            shutil.move(input_path, output_path)

print("Moving images to train_data completed!")

images_root_path = '/home/wallace/Code/HUST/Deep_Learning/OCR/resources/train/data'
output_file_path = 'all_train_images.txt'

with open(output_file_path, 'w') as output_file:
    for root, dirs, files in os.walk(images_root_path):
        for file in files:
            if file.lower().endswith('.jpg'):
                output_file.write(file + '\n')

print("all.txt file created!")

input_file_path = 'all_train_images.txt'
train_output_file_path = '../train_inds.pkl'
val_output_file_path = '../val_inds.pkl'

# Set the train/validation split ratio
train_ratio = 0.7

with open(input_file_path, 'r') as input_file:
    lines = input_file.readlines()

random.shuffle(lines)
num_lines = len(lines)
num_train_lines = int(num_lines * train_ratio)
num_val_lines = num_lines - num_train_lines
train_lines = lines[:num_train_lines]
val_lines = lines[num_train_lines:]

train_image_names = [line.strip() for line in train_lines]
val_image_names = [line.strip() for line in val_lines]

with open(train_output_file_path, 'wb') as train_output_file:
    pickle.dump(train_image_names, train_output_file)
with open(val_output_file_path, 'wb') as val_output_file:
    pickle.dump(val_image_names, val_output_file)

print("train_inds.pkl and val_inds.pkl files created!")