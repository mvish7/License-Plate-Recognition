# License Plate Recognition
 
Here you can find code of the attempt I made to implement an (Automatic) license plate recognition. This implemented is based on Connected Component Analysis, 
Optical Character Recognition and Character classification.

## Connected Component Analysis: CCA finds the connected regions from the images, connected pixels are the once that share the same value and are adjacent to each other.
The connected region is formed by collecting all such pixels.

# Process

Following steps were taken to realize the license plate recognition

## Image preprocessing: 
The image was loaded on gray scale and then thresholding was applied to create a binary image. This binary image helps us finding connected components.

## Finding license plate:

The connected component analysis is applied over the binary image to find connected regions. At the first level, the regions having less area than a specific value were omitted.
In the second level, dimensions of the license plate were assumed (e.g. width of a license plate should be greater than its height) and regions not following this criterion were eliminated.
The remaining regions can be classified as plate-like objects, we are yet to select an exact license plate from remaining connected regions.

### Identifying license plate from the plate-like objects:

To find the exact license plate from the plate-like objects, I tried to use 3 different techniques. None of them work satisfactorily.

* Vertical Projection of Pixels: It was based on the assumption that the sum of all the pixels from a region representing license plate should be higher than other connected regions.

* Connected Components inside connected regions: It was based on the assumption that the license plate would have more connected regions than other candidates as the characters in the license plate
would represent a connected region.

* Finding Number of Contours in regions: Here the assumption was made that a license plate should have more contours object in it than other candidates. 

## Character Segmention:

To segment character, the license plate was selected manually from the remaining candidate (That's why it's not a fully automatic recognition). Some assumptions were made about the dimensions of the characters in the license plate. The connected regions were found once again and the dimensions of the regions compared against dimensions of the characters to filter out non-character regions.

## Character Classification:

For classification, I trained an SVC with the help of sklearn and I created a logistic regression model in Pytorch. The data of A-Z letters and 0-9 digits were used as training data. The training data can be found [here](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/)
for predicting the license plate characters, separate scripts were made for SVC and torch models.

## ToDos:

* Improve the logic for finding the license plate amongst the candidates
* Setup a testing and evaluation mechanism
