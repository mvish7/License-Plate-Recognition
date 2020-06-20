from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.measure import regionprops
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class LicensePlate:
    def __init__(self, image_filename):
        self.image_filename = image_filename
        self.binary_image = None  # output of process_image_function
        self.plate_like_objects = []  # license plate candidates
        self.plate_objects_cordinates = []

    def process_image(self, viz=False):
        """
        Functions reads the images, converts it to grayscale and applies thresholding
        :return:
        """

        query_image = imread(self.image_filename, as_grey=True)
        # print(query_image.shape)

        self.gray_image = query_image * 255
        threshold = threshold_otsu(self.gray_image)
        self.binary_image = self.gray_image > threshold

        # controlling the visualization
        if viz:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(self.gray_image, cmap="gray")
            ax2.imshow(self.binary_image, cmap="gray")
            plt.show()

    def locate_license_plate(self):
        """
        Function finds connected components in image which forms basis for license plate like objects.
        then these connected components are filtered on the basis of their dimensions

        Connected components : A pixel is deemed to be connected to another if they both have the same value and are
        adjacent to each other.

        some assumptions about license plate shape:
        width > height
        width occupies upto 20-40% area of the plate
        height occupies upto 10-20% area of the plate
        :return:
        """

        # plate_like_objects = []
        # plate_objects_cordinates = []

        # all the connected regions are being found and grouped together
        label_image = measure.label(self.binary_image)

        # assuming the maximum width, height and minimum width and height of a license plate
        plate_dimensions = (0.08 * label_image.shape[0], 0.2 * label_image.shape[0], 0.15 * label_image.shape[1],
                            0.4 * label_image.shape[1])
        min_height, max_height, min_width, max_width = plate_dimensions

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(self.gray_image, cmap="gray")

        # regionprops creates a list of properties of all the labelled regions
        for region in regionprops(label_image):
            if region.area < 50:
                # if a region has area smaller than let's say 50 then it's not likely to be a license plate
                continue

            # the bounding box coordinates
            minX, minY, maxX, maxY = region.bbox

            region_height = maxX - minX
            region_width = maxY - minY

            if region_width > region_height:
                if(region_height >= min_height and region_height <= max_height)\
                        and (region_width >= min_width and region_width <= max_width):

                    self.plate_like_objects.append(self.binary_image[minX:maxX, minY:maxY])
                    self.plate_objects_cordinates.append((minX, minY, maxX, maxY))
                    rectBorder = patches.Rectangle((minY, minX), maxY - minY, maxX - minX, edgecolor="blue",
                                                   linewidth=1, fill=False)
                    ax1.add_patch(rectBorder)

        # trying to select exact license plate, based on the assumption that sum of pixel values of a license plate
        # should be greater  than plate like object when summed column wise -------- doesn't work well.

        # license_plate = []
        # highest_sum = 0
        # selected_plate = None
        # ax2.imshow(self.gray_image, cmap="gray")
        # for i, candidate in enumerate(plate_like_objects):
        #     height, width = candidate.shape
        #
        #     #  inverting the plate candidate as license plate has more of whites and they will have higher gray scale
        #     # value than character regions so inverting
        #     threshold = threshold_otsu(self.gray_image)
        #     candidate = self.gray_image < threshold
        #
        #     total_white_pixels = 0
        #     for column in range(width):
        #         total_white_pixels += sum(candidate[:, column])
        #
        #     if i == 0:
        #         highest_sum = total_white_pixels
        #
        #     if total_white_pixels >= highest_sum:
        #         license_plate = candidate
        #         selected_plate = i
        #
        # plate_rect = plate_objects_cordinates[selected_plate]
        # plateBorder = patches.Rectangle((plate_rect[1], plate_rect[0]), plate_rect[3] - plate_rect[1], plate_rect[2] - plate_rect[0],
        #                                 edgecolor="blue", linewidth=1, fill=False)
        # ax2.add_patch(plateBorder)
        # plt.show()

        # ##### assumption - candidate with highest connected components can be the
        # license plate --------- doesn't work well.

        # ax2.imshow(self.gray_image, cmap="gray")
        # connected_regions = []
        # for i, candidate in enumerate(plate_like_objects):
        #     connected_regions.append(len(measure.label(candidate)))
        #
        # connected_regions = np.asarray(connected_regions, dtype=np.int)
        # license_plate_id = np.argmax(connected_regions)
        #
        # plate_rect = plate_objects_cordinates[license_plate_id]
        # plateBorder = patches.Rectangle((plate_rect[1], plate_rect[0]), plate_rect[3] - plate_rect[1], plate_rect[2] - plate_rect[0],
        #                                 edgecolor="blue", linewidth=1, fill=False)
        # ax2.add_patch(plateBorder)
        # plt.show()

        # ######### assumption - candidate with max countours can be license plate

        # return license_plate

    def segment_characters(self):
        """
        function segments each characters on the license plate using connected component analysis
        :param license_plate:
        :return:
        """

        # The invert was done so as to convert the black pixel to white pixel and vice versa
        license_plate = np.invert(self.plate_like_objects[2])

        labelled_plate = measure.label(license_plate)

        fig, ax1 = plt.subplots(1)
        ax1.imshow(license_plate, cmap="gray")

        # again using assumptions of license plate shape to define character dimension bound

        character_dimensions = (0.35 * license_plate.shape[0], 0.60 * license_plate.shape[0],
                                 0.05 * license_plate.shape[1], 0.15 * license_plate.shape[1])

        min_height, max_height, min_width, max_width = character_dimensions

        characters = []
        char_seq_lst = []
        for regions in regionprops(labelled_plate):
            y0, x0, y1, x1 = regions.bbox
            region_height = y1 - y0
            region_width = x1 - x0

            if region_height > min_height and region_height < max_height \
                    and region_width > min_width and region_width < max_width:
                roi = license_plate[y0:y1, x0:x1]

                # draw a red bordered rectangle over the character.
                rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",
                                                linewidth=1, fill=False)
                ax1.add_patch(rect)

                # resize the characters to 28X28 as our training images are 28X28 and then append each character into
                #  the characters list
                resized_characters = resize(roi, (28, 28))
                characters.append(resized_characters)

                # this is just to keep track of the arrangement of the characters, storing the start of the bounding box
                #  of each character, so we can sort them in the end and rearrange
                char_seq_lst.append(x0)

        plt.show()
        return characters, char_seq_lst




# ############### test section

def main():
    image_filename = "C:/Users/mvish/PycharmProjects/ALPR/test_images/car6.jpg"
    license_plate_recog = LicensePlate(image_filename)

    license_plate_recog.process_image(viz=True)
    license_plate_recog.locate_license_plate()
    character_lst, char_seq_lst = license_plate_recog.segment_characters()


if __name__ == "__main__":
    main()