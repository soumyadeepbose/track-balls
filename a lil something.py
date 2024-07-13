import cv2
import glob
import os
        
def normalize_box_coordinates(folder_path_images, folder_path_labels):

    # Pattern to match all .txt files in the labels folder
    pattern = os.path.join(folder_path_labels, '*.txt')

    text_files = glob.glob(pattern)

    for label_file in text_files:
        # Deriving the corresponding image file path
        base_name = os.path.basename(label_file)
        image_file = os.path.join(folder_path_images, base_name.replace('.txt', '.jpg'))  # Assuming images are .jpg

        image = cv2.imread(image_file)
        height, width, _ = image.shape
        
        with open(label_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Normalizing box coordinates
        normalized_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # format is: class x_center y_center box_width box_height
                x1 = float(parts[1]) #/ width
                y1 = float(parts[2]) #/ height
                x2 = float(parts[3]) #/ width
                y2 = float(parts[4]) #/ height

                normalized_line = f"{0} {float((x1+x2)/2)} {float((y1+y2)/2)} {float(x2-x1)} {float(y2-y1)}\n"
                print(normalized_line)

                # Replacing 'Ball' with '0'
                # normalized_line = content.replace('Ball', '0')

                normalized_lines.append(normalized_line)

        with open(label_file, 'w', encoding='utf-8') as file:
            file.writelines(normalized_lines)
            

folder_path_images = 'ballz/train/images'
folder_path_labels = 'ballz/train/labels'
normalize_box_coordinates(folder_path_images, folder_path_labels)
print("Train Normalization Done!")

folder_path_images = 'ballz/val/images'
folder_path_labels = 'ballz/val/labels'
normalize_box_coordinates(folder_path_images, folder_path_labels)
print("Val Normalization Done!")