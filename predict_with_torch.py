import torch
import license_plate_recognition as lp_recog
import torch_classifier


def main():

    model = torch_classifier.Net(is_training=False)
    model.load_state_dict(torch.load('C:\\Users\\mvish\\PycharmProjects\\AutomaticLicensePlateRecognition\\model\\torch_classifier.pth'))

    image_filename = "C:/Users/mvish/PycharmProjects/ALPR/test_images/car6.jpg"
    license_plate_recog = lp_recog.LicensePlate(image_filename)

    license_plate_recog.process_image(viz=True)
    license_plate_recog.locate_license_plate()
    character_lst, char_seq_lst = license_plate_recog.segment_characters()
    result = []
    for char in character_lst:
        each_char = torch.tensor(char, dtype=torch.float32)

        pred = model.infer(each_char)
        pred = pred.indices.item()
        # print(pred)

        # form a dict of labels and their mearning

        op_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B',
                   12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M',
                   23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X',
                   34: 'Y', 35: 'Z'}

        result.append(op_dict[pred])

    # from each pred.indices find the output charcater by indexing the dict
    # correct the order of predictions same as svc

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
