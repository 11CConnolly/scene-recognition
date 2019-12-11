package ecs.comp3204.cw3.run1;

import org.openimaj.feature.FloatFV;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;

public class TinyImage {

    FImage originalImage;
    FImage squaredOriginal;
    FImage tinyImage;

    public TinyImage(FImage orginalImage, int tinyWidth) {

        this.originalImage = orginalImage;
        int sqLength;

        // get the length for squaredOriginal
        if (originalImage.height > originalImage.width) {
            sqLength = orginalImage.width;
        } else {
            sqLength = originalImage.height;
        }

        // create a temp to copy the pixels
        float[][] pixels = new float [sqLength][sqLength];

        // trim the image to a square if it is not already a square
        for (int i = 0; i < sqLength; i++) {
            for (int j = 0; j < sqLength; j++) {
                pixels[i][j] = originalImage.pixels[i][j];
            }
        }

        // create a squared version of the original image
        squaredOriginal = new FImage(pixels);

        // resample the image to a tiny image of the desired width
        tinyImage = ResizeProcessor.resample(new FImage(pixels), tinyWidth, tinyWidth);


        // Need to determine max and min value for the ratio
        // the sum of the pixel values divided by the number of pixels = mean value for each pixel

        float largest = Float.MIN_VALUE;
        float smallest = Float.MAX_VALUE;
        float total = 0;

        for (float[] row : tinyImage.pixels) {
            for (float pixel : row) {
                if (pixel > largest) {
                    largest = pixel;
                } else if (pixel < smallest) {
                    smallest = pixel;
                }
                total += pixel;
            }
        }

        float mean = (total/(tinyWidth*tinyWidth));

        for (int i = 0; i < tinyImage.height; i++) {
            for (int j = 0; j < tinyImage.width; j++) {
                // (- mean) to reduce to zero mean
                // (* ratio) to make unit length
                tinyImage.pixels[i][j] = (tinyImage.pixels[i][j] - mean) * (1/(largest-smallest));
            }
        }
    }

    // get vector value of the image
    public FloatFV getImageVector() {

        FloatFV fv;
        float[] vector = new float[tinyImage.height*tinyImage.width];

        for (int i = 0; i < tinyImage.height; i++) {
            for(int j = 0; j < tinyImage.width; j++) {
                vector[i*tinyImage.width+j] = tinyImage.pixels[i][j];
            }
        }
        fv = new FloatFV(vector);

        return fv;
    }
}
