from google.colab import drive
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage import img_as_float
import pandas as pd
import numpy as np
from sklearn import metrics

def calculate_metrics(data_csv_path, folder_path):
    # Mount Google Drive
    drive.mount('/content/drive', force_remount=True)

    # Read CSV file
    df = pd.read_csv(data_csv_path)

    # Filter rows where Prediction contains 'Anomaly'
    df = df[df['Prediction'].str.contains('Anomaly')]

    # Reflect functions
    def reflect_x(x, y, origin_x, origin_y):
        return (2 * origin_x - x, y)

    def reflect_y(x, y, origin_x, origin_y):
        return (x, 2 * origin_y - y)

    def reflect_origin(x, y, origin_x, origin_y):
        return (2 * origin_x - x, 2 * origin_y - y)

    # Apply reflection functions
    df['X0_reflect_x'], df['Y0_reflect_x'] = reflect_x(df['X_0'], df['Y_0'], 112, 112)
    df['X1_reflect_x'], df['Y1_reflect_x'] = reflect_x(df['X_1'], df['Y_1'], 112, 112)

    df['X0_reflect_y'], df['Y0_reflect_y'] = reflect_y(df['X_0'], df['Y_0'], 112, 112)
    df['X1_reflect_y'], df['Y1_reflect_y'] = reflect_y(df['X_1'], df['Y_1'], 112, 112)

    df['X0_reflect_origin'], df['Y0_reflect_origin'] = reflect_origin(df['X_0'], df['Y_0'], 112, 112)
    df['X1_reflect_origin'], df['Y1_reflect_origin'] = reflect_origin(df['X_1'], df['Y_1'], 112, 112)

    # Find quadrant functions
    def find_quadrant(X_0, Y_0, X_1, Y_1):
        centroid_x = (X_0 + X_1) / 2
        centroid_y = (Y_0 + Y_1) / 2

        if centroid_x > 112:
            return 1 if centroid_y < 112 else 4
        else:
            return 2 if centroid_y < 112 else 3

    def find_quadrant2(df):
        df['centroid_x'] = (df['X_0'] + df['X_1']) / 2
        df['centroid_y'] = (df['Y_0'] + df['Y_1']) / 2

        quadrants = []

        for _, row in df.iterrows():
            if row['centroid_x'] > 112:
                quadrants.append(1) if row['centroid_y'] < 112 else quadrants.append(4)
            else:
                quadrants.append(2) if row['centroid_y'] < 112 else quadrants.append(3)

        df['quadrant'] = quadrants
        return df

    # Apply find_quadrant2 function
    df = find_quadrant2(df)

    # Get subarray function
    def get_subarray(image_path, X_0, Y_0, X_1, Y_1):
        X_0, X_1 = min(X_0, X_1), max(X_0, X_1)
        Y_0, Y_1 = min(Y_0, Y_1), max(Y_0, Y_1)

        img = Image.open(image_path)
        subarray = img.crop((X_0, Y_0, X_1, Y_1))
        return subarray

    # Initialize SSIM dictionary
    ssim_dict = {}

    for index, row in df.iterrows():
        image_path = f'{folder_path}/subplot_{row["Image"]}.png'
        subarrays = [
            (get_subarray(image_path, row['X_0'], row['Y_0'], row['X_1'], row['Y_1']), (row['X_0'], row['Y_0'], row['X_1'], row['Y_1'])),
            (get_subarray(image_path, row['X0_reflect_x'], row['Y0_reflect_x'], row['X1_reflect_x'], row['Y1_reflect_x']), (row['X0_reflect_x'], row['Y0_reflect_x'], row['X1_reflect_x'], row['Y1_reflect_x'])),
            (get_subarray(image_path, row['X0_reflect_y'], row['Y0_reflect_y'], row['X1_reflect_y'], row['Y1_reflect_y']), (row['X0_reflect_y'], row['Y0_reflect_y'], row['X1_reflect_y'], row['Y1_reflect_y'])),
            (get_subarray(image_path, row['X0_reflect_origin'], row['Y0_reflect_origin'], row['X1_reflect_origin'], row['Y1_reflect_origin']), (row['X0_reflect_origin'], row['Y0_reflect_origin'], row['X1_reflect_origin'], row['Y1_reflect_origin']))
        ]

        ssim_df = pd.DataFrame()

        for i in range(len(subarrays)):
            for j in range(i+1, len(subarrays)):
                subarray_1, points_1 = subarrays[i]
                subarray_2, points_2 = subarrays[j]

                quadrant_1 = find_quadrant(*points_1)
                quadrant_2 = find_quadrant(*points_2)

                if quadrant_1 == 1:
                    if quadrant_2 == 2:
                        subarray_2 = np.fliplr(subarray_2)
                    elif quadrant_2 == 3:
                        subarray_2 = np.flipud(np.fliplr(subarray_2))
                    elif quadrant_2 == 4:
                        subarray_2 = np.flipud(subarray_2)

                if quadrant_1 == 2:
                    if quadrant_2 == 1:
                        subarray_2 = np.fliplr(subarray_2)
                    elif quadrant_2 == 4:
                        subarray_2 = np.flipud(np.fliplr(subarray_2))
                    elif quadrant_2 == 3:
                        subarray_2 = np.flipud(subarray_2)

                if quadrant_1 == 3:
                    if quadrant_2 == 4:
                        subarray_2 = np.fliplr(subarray_2)
                    elif quadrant_2 == 1:
                        subarray_2 = np.flipud(np.fliplr(subarray_2))
                    elif quadrant_2 == 2:
                        subarray_2 = np.flipud(subarray_2)

                if quadrant_1 == 4:
                    if quadrant_2 == 3:
                        subarray_2 = np.fliplr(subarray_2)
                    elif quadrant_2 == 2:
                        subarray_2 = np.flipud(np.fliplr(subarray_2))
                    elif quadrant_2 == 1:
                        subarray_2 = np.flipud(subarray_2)

                img1 = img_as_float(subarray_1)
                img2 = img_as_float(subarray_2)

                ssim = compare_ssim(img1, img2, channel_axis=-1)

                ssim_df.loc[quadrant_1, quadrant_2] = ssim
                ssim_df.loc[quadrant_2, quadrant_1] = ssim

        ssim_df.sort_index(axis=0, inplace=True)
        ssim_df.sort_index(axis=1, inplace=True)
        ssim_df.fillna(1, inplace=True)

        ssim_dict[row["Image"]] = ssim_df

    min_mean_df = pd.DataFrame(columns=['Image', 'Min_Mean_Row', 'Min_Mean'])

    for image, ssim_df in ssim_dict.items():
        means = ssim_df.mean(axis=1)
        min_mean_row = means.idxmin()
        min_mean = means.min()

        min_mean_df = min_mean_df.append({'Image': image, 'Min_Mean_Row': min_mean_row, 'Min_Mean': min_mean}, ignore_index=True)

    ground_truth = min_mean_df['Min_Mean_Row'].values
    predicted_val = df['quadrant'].values

    # Calculate metrics
    accuracy = metrics.accuracy_score(ground_truth, predicted_val)
    balanced_accuracy = metrics.balanced_accuracy_score(ground_truth, predicted_val)
    precision = metrics.precision_score(ground_truth, predicted_val, average='weighted')
    recall = metrics.recall_score(ground_truth, predicted_val, average='weighted')
    f1 = metrics.f1_score(ground_truth, predicted_val, average='weighted')

    # Print metrics
    # print(f'Accuracy: {accuracy}')
    # print(f'Balanced Accuracy: {balanced_accuracy}')
    # print(f'Precision: {precision}')
    # print(f'Recall: {recall}')
    # print(f'F1 Score: {f1}')
    return accuracy

# data_csv_path = '/content/DamageDetection/data.csv'
# calculate_metrics(data_csv_path)
