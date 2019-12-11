package ecs.comp3204.cw3.run1;

import org.apache.commons.vfs2.FileSystemException;
import org.apache.commons.vfs2.impl.StandardFileSystemManager;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.typography.hershey.HersheyFont;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.Random;

/**
 * Run1
 *
 */
public class App {

    // k value for the classifier
    private final static int k = 28;

    public static void main (String[] args) throws Exception {

        // Setup a new classifier
        KNNClassifier classifier = new KNNClassifier();

        // input data from url directly
        //VFSListDataset<FImage> testingSet = new VFSListDataset<FImage>("zip:http://comp3204.ecs.soton.ac.uk/cw/testing.zip",ImageUtilities.FIMAGE_READER);
        VFSGroupDataset<FImage> trainingSet = new VFSGroupDataset<>("zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip", ImageUtilities.FIMAGE_READER);
        // the group named "training" contains all the images in the zip, we do not need it!
        trainingSet.remove("training");
        System.out.println("Number of Categories: " + (trainingSet.size()));
        //add the training data to the classifier
        classifier.addTrainingData(trainingSet, k);


        // I WILL FIX LATER LET ME SLEEP
//        File folder = new File("./data/testing");
//        File[] listOfFiles = folder.listFiles();
//        for (File file : listOfFiles) {
//            if (file.isFile()) {
//                classifier.classify(new FImage(file),k);
//            }
//        }
    }


}
