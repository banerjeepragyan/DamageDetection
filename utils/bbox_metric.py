import pandas as pd
import numpy as np

def reflect_x(x, y, origin_x, origin_y):
    return (2*origin_x - x, y)

def reflect_y(x, y, origin_x, origin_y):
    return (x, 2*origin_y - y)

def reflect_origin(x, y, origin_x, origin_y):
    return (2*origin_x - x, 2*origin_y - y)

def get_subarray(image_path, X_0, Y_0, X_1, Y_1):
    X_0, X_1 = min(X_0, X_1), max(X_0, X_1)
    Y_0, Y_1 = min(Y_0, Y_1), max(Y_0, Y_1)

    img = Image.open(image_path)

    subarray = img.crop((X_0, Y_0, X_1, Y_1))
    return subarray

from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
from PIL import Image

def bbox_metric(file_path, image_folder):

# file_path = '/content/drive/MyDrive/data.csv' #Enter the path of the csv file
    df = pd.read_csv(file_path)

    df = df[df['Prediction'].str.contains('Anomaly')]

    df['X0_reflect_x'], df['Y0_reflect_x'] = reflect_x(df['X_0'], df['Y_0'], 112, 112)
    df['X1_reflect_x'], df['Y1_reflect_x'] = reflect_x(df['X_1'], df['Y_1'], 112, 112)

    df['X0_reflect_y'], df['Y0_reflect_y'] = reflect_y(df['X_0'], df['Y_0'], 112, 112)
    df['X1_reflect_y'], df['Y1_reflect_y'] = reflect_y(df['X_1'], df['Y_1'], 112, 112)

    df['X0_reflect_origin'], df['Y0_reflect_origin'] = reflect_origin(df['X_0'], df['Y_0'], 112, 112)
    df['X1_reflect_origin'], df['Y1_reflect_origin'] = reflect_origin(df['X_1'], df['Y_1'], 112, 112)

    ssim_dict = {}

    for index, row in df.iterrows():
        image_path = f'{image_folder}/subplot_{row["Image"]}.png' #Enter the path of image file
        subarray_1 = get_subarray(image_path, row['X_0'], row['Y_0'], row['X_1'], row['Y_1'])
        subarray_2 = get_subarray(image_path, row['X0_reflect_x'], row['Y0_reflect_x'], row['X1_reflect_x'], row['Y1_reflect_x'])
        subarray_3 = get_subarray(image_path, row['X0_reflect_y'], row['Y0_reflect_y'], row['X1_reflect_y'], row['Y1_reflect_y'])
        subarray_4 = get_subarray(image_path, row['X0_reflect_origin'], row['Y0_reflect_origin'], row['X1_reflect_origin'], row['Y1_reflect_origin'])

        img1 = img_as_float(subarray_1)
        img2 = img_as_float(subarray_2)
        img3 = img_as_float(subarray_3)
        img4 = img_as_float(subarray_4)


        ssim_df = pd.DataFrame(index=['original', 'X axis reflect', 'Y axis reflect', 'Origin Reflected'],
                            columns=['original', 'X axis reflect', 'Y axis reflect', 'Origin Reflected'])


        ssim_df.loc['original', 'original'] = 1
        ssim_df.loc['original', 'X axis reflect'] = ssim(img1, img2, multichannel=True)
        ssim_df.loc['original', 'Y axis reflect'] = ssim(img1, img3, multichannel=True)
        ssim_df.loc['original', 'Origin Reflected'] = ssim(img1, img4, multichannel=True)
        ssim_df.loc['X axis reflect', 'original'] = ssim_df.loc['original', 'X axis reflect']
        ssim_df.loc['X axis reflect', 'X axis reflect'] = 1
        ssim_df.loc['X axis reflect', 'Y axis reflect'] = ssim(img2, img3, multichannel=True)
        ssim_df.loc['X axis reflect', 'Origin Reflected'] = ssim(img2, img4, multichannel=True)
        ssim_df.loc['Y axis reflect', 'original'] = ssim_df.loc['original', 'Y axis reflect']
        ssim_df.loc['Y axis reflect', 'X axis reflect'] = ssim_df.loc['X axis reflect', 'Y axis reflect']
        ssim_df.loc['Y axis reflect', 'Y axis reflect'] = 1
        ssim_df.loc['Y axis reflect', 'Origin Reflected'] = ssim(img3, img4, multichannel=True)
        ssim_df.loc['Origin Reflected', 'original'] = ssim_df.loc['original', 'Origin Reflected']
        ssim_df.loc['Origin Reflected', 'X axis reflect'] = ssim_df.loc['X axis reflect', 'Origin Reflected']
        ssim_df.loc['Origin Reflected', 'Y axis reflect'] = ssim_df.loc['Y axis reflect', 'Origin Reflected']
        ssim_df.loc['Origin Reflected', 'Origin Reflected'] = 1

        ssim_dict[row["Image"]] = ssim_df


    # Initialize a new dataframe to store the rows with the minimum mean
    min_mean_df = pd.DataFrame(columns=['Image', 'Min_Mean_Row', 'Min_Mean'])

    for image, ssim_df in ssim_dict.items():
        means = ssim_df.mean(axis=1)

        min_mean_row = means.idxmin()
        min_mean = means.min()

        min_mean_df = min_mean_df.append({'Image': image, 'Min_Mean_Row': min_mean_row, 'Min_Mean': min_mean}, ignore_index=True)


    quadrants = []

    for _, row in df.iterrows():
        center_x, center_y = row['X_max'] / 2, row['Y_max'] / 2
        centroid_x = (row['X_0'] + row['X_1']) / 2
        centroid_y = (row['Y_0'] + row['Y_1']) / 2

        if centroid_x < center_x and centroid_y < center_y:
            quadrant = 2
        elif centroid_x >= center_x and centroid_y < center_y:
            quadrant = 1
        elif centroid_x < center_x and centroid_y >= center_y:
            quadrant = 3
        else:
            quadrant = 4

        quadrants.append(quadrant)

    quadrants_df = pd.DataFrame(quadrants, columns=['Quadrant'])

    quadrants_min_mean_df = []

    for _, row in min_mean_df.iterrows():
        image = row['Image']
        min_mean_row = row['Min_Mean_Row']

        if min_mean_row == 'original':
            gt_box = df.loc[df['Image'] == image, ['X_0', 'Y_0', 'X_1', 'Y_1']].values[0]
        elif min_mean_row == 'X axis reflect':
            gt_box = df.loc[df['Image'] == image, ['X0_reflect_x', 'Y0_reflect_x', 'X1_reflect_x', 'Y1_reflect_x']].values[0]
        elif min_mean_row == 'Y axis reflect':
            gt_box = df.loc[df['Image'] == image, ['X0_reflect_y', 'Y0_reflect_y', 'X1_reflect_y', 'Y1_reflect_y']].values[0]
        else:  # 'Origin Reflected'
            gt_box = df.loc[df['Image'] == image, ['X0_reflect_origin', 'Y0_reflect_origin', 'X1_reflect_origin', 'Y1_reflect_origin']].values[0]

        centroid_x = (gt_box[0] + gt_box[2]) / 2
        centroid_y = (gt_box[1] + gt_box[3]) / 2

        if centroid_x < center_x and centroid_y < center_y:
            quadrant = 2
        elif centroid_x >= center_x and centroid_y < center_y:
            quadrant = 1
        elif centroid_x < center_x and centroid_y >= center_y:
            quadrant = 3
        else:  # centroid_x >= center_x and centroid_y >= center_y
            quadrant = 4

        quadrants_min_mean_df.append(quadrant)

    quadrants_min_mean_df = pd.DataFrame(quadrants_min_mean_df, columns=['Quadrant'])

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
    quadrants_min_mean_df_array = quadrants_min_mean_df['Quadrant'].to_numpy()
    quadrants_array = np.array(quadrants)

    accuracy = accuracy_score(quadrants_min_mean_df_array, quadrants_array)
    precision = precision_score(quadrants_min_mean_df_array, quadrants_array, average='macro')
    recall = recall_score(quadrants_min_mean_df_array, quadrants_array, average='macro')
    f1 = f1_score(quadrants_min_mean_df_array, quadrants_array, average='macro')
    balanced_accuracy = balanced_accuracy_score(quadrants_min_mean_df_array, quadrants_array)

    # print("Accuracy is: ", accuracy)
    # print("Precision is: ", precision)
    # print("Recall score is: ", recall)
    # print("F1 score is: ", f1)
    # print("Balanced Accuracy is: ", balanced_accuracy)

    return accuracy, precision, recall, f1, balanced_accuracy