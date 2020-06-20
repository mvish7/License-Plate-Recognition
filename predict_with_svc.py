import os
from sklearn.externals import joblib
import license_plate_recognition as lp_recog


def main():
    # current_dir = os.path.dirname(os.path.realpath(__file__))
    # model_dir = os.path.join(current_dir, 'model/svc/scv.pkl')
    model_dir = 'C:\\Users\\mvish\\PycharmProjects\\AutomaticLicensePlateRecognition\\model\\svc\\svc.pkl'
    model = joblib.load(model_dir)

    image_filename = "C:/Users/mvish/PycharmProjects/ALPR/test_images/car6.jpg"
    license_plate_recog = lp_recog.LicensePlate(image_filename)

    license_plate_recog.process_image(viz=True)
    license_plate_recog.locate_license_plate()
    character_lst, char_seq_lst = license_plate_recog.segment_characters()

    result = []
    for char in character_lst:
        # converts it to a 1D array
        single_char = char.reshape(1, -1)
        pred = model.predict(single_char)
        result.append(pred)

    # print(result)

    plate_string = ''
    for entry in result:
        plate_string += entry[0]

    print(plate_string)

    # characters are being printed with wrong sequence, keep the track of sequence of characters while segmenting

    # rearranging the characters

    char_lst_copy = char_seq_lst[:]
    char_seq_lst.sort()
    correct_string = ''
    for each in char_seq_lst:
        correct_string += plate_string[char_lst_copy.index(each)]

    print(correct_string)



if __name__ == "__main__":
    main()
