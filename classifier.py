import os
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from skimage.filters import threshold_otsu
from skimage.transform import rotate, pyramid_reduce
from PIL import Image
import numpy as np
from sklearn import metrics


def read_n_process_training_data():

    labels_set = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
        'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
        'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ]

    data_dir = 'data/'
    images = []
    labels = []

    img_dirs = os.listdir(data_dir)
    for i, tmp_dir in enumerate(img_dirs):
        curr_dir_imgs = os.listdir(os.path.join(data_dir, tmp_dir))

        for j, img in enumerate(curr_dir_imgs):
            img_path = os.path.join(data_dir, tmp_dir, curr_dir_imgs[j])
            img_data = Image.open(img_path).convert('L')
            img_data = np.array(img_data.resize((28, 28)))
            binary_img = img_data < threshold_otsu(img_data)


            # flattening to create 1d array for our models

            flat_bin_img = binary_img.reshape(-1)
            images.append(flat_bin_img)
            labels.append(labels_set[i])

    return np.array(images), np.array(labels)


def main():

    images, labels = read_n_process_training_data()

    clf = SVC(kernel='linear', probability=True)

    # cross validation

    accuracy = cross_val_score(clf, images, labels, cv=4)
    print("Cross Validation Result for ", str(4), " -fold")
    print(accuracy * 100)

    clf.fit(images, labels)

    save_directory = os.path.join('C:\\Users\\mvish\\PycharmProjects\\AutomaticLicensePlateRecognition\\', 'model/svc/')
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    joblib.dump(clf, save_directory + '/svc.pkl')

if __name__ == "__main__":
    main()