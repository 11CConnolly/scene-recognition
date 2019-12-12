package ecs.comp3204.cw3.run2;

import org.apache.commons.io.comparator.NameFileComparator;
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

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Run1
 *
 */
public class App {

    // patch size is 8*8
    private final static int patchSize = 8;
    // patch distance
    private final static int patchDistance = 8;
    // ~500 clusters as said in spec
    private final static int clusters = 500;

    public static void main (String[] args) throws Exception {

        System.out.println("This is run 2!");

        // Setup a new classifier
        LinearClassifier classifier = new LinearClassifier(patchSize, patchDistance, clusters);

        // input data from url directly
        VFSGroupDataset<FImage> trainingSet = new VFSGroupDataset<>("zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip", ImageUtilities.FIMAGE_READER);
        // the group named "training" contains all the images in the zip, we do not need it!
        trainingSet.remove("training");
        System.out.println("Number of Categories provided: " + (trainingSet.size()));
        //add the training data to the classifier
        classifier.addTrainingData(trainingSet);

        File folder = new File("./data/testing");
        File[] files = folder.listFiles();
        // sort the outputs in numberic order instead of 1, 10, 100...
        Arrays.sort(files, new Comparator<File>() {
            @Override
            public int compare(File o1, File o2) {
                // get filename of files e.g. "1.jpg"
                String f1 = o1.getName();
                String f2 = o2.getName();

                // remove extensions ".jpg"
                f1 = f1.substring(0, f1.indexOf("."));
                f2 = f2.substring(0, f2.indexOf("."));

                int int1 = Integer.parseInt(f1);
                int int2 = Integer.parseInt(f2);

                // return compared results
                return int1-int2;
            }
        });

        String pathname = "run2.txt";
        File run1output = new File(pathname);
        if (!run1output.exists()) {
            run1output.createNewFile();
        }

        // writing to file "run1.txt"
        BufferedWriter bw = new BufferedWriter(new FileWriter(run1output));

        for (File file : files) {
            if (file.isFile()) {
                FImage fi = ImageUtilities.readF(file);
                String guess = classifier.classify(fi);
                String output = file.getName()+" "+guess+"\n";
                //System.out.println(output);
                bw.write(output);
            }
        }
        System.out.println("Successfully written output to file " + pathname + "!");
        bw.close();
    }


}
