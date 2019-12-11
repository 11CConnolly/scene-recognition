package ecs.comp3204.cw3.run2;


import ecs.comp3204.cw3.run1.TinyImage;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.FloatFV;
import org.openimaj.image.FImage;

import java.util.HashSet;
import java.util.Map;

public class LinearClassifier {

    int patchSize;
    int numPatches;
    int clusters;

    HashSet<FloatFV> fv = new HashSet<>();

    // Default Setting
    public LinearClassifier() {
        this.patchSize = 8;
        this.numPatches = 10;
        this.clusters = 500;
    }

    public LinearClassifier(int patchSize, int numPatches, int clusters) {
        this.patchSize = patchSize;
        this.numPatches = numPatches;
        this.clusters = clusters;
    }


    // compare image with the testing data, returns a guess
    public String classify(FImage fi) {
        String guess = null;

        return guess;
    }

    // create patches
    public void addTrainingData(VFSGroupDataset<FImage> data) {
        // separating the data into categories
        for (final Map.Entry<String, VFSListDataset<FImage>> entry : data.entrySet()) {
            for (FImage img : entry.getValue()) {
                // something something patches related
            }
        }
    }


}
