package ecs.comp3204.cw3;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.typography.hershey.HersheyFont;

/**
 * OpenIMAJ Hello world!
 *
 */
public class App {
    public static void main( String[] args ) {
    	//Create an image
        MBFImage image = new MBFImage(888,200, ColourSpace.RGB);

        //Fill the image with white
        image.fill(RGBColour.BLACK);
        		        
        //Render some test into the image
        image.drawText("Hello World! We are Team 33!", 10, 60, HersheyFont.TIMES_BOLD, 50, RGBColour.RED);
        image.drawText("Callum Connolly", 10, 140, HersheyFont.CURSIVE, 30, RGBColour.RED);
        image.drawText("Kwok Hung Liu", 10, 170, HersheyFont.CURSIVE, 30, RGBColour.RED);
        image.drawText("Peter Deng", 10, 200, HersheyFont.CURSIVE, 30, RGBColour.RED);

        //Apply a Gaussian blur - nah
        image.processInplace(new FGaussianConvolve(0f));
        
        //Display the image
        DisplayUtilities.display(image);
    }
}
