package ecs.comp3204.cw3.run2;


import de.bwaldvogel.liblinear.SolverType;
import ecs.comp3204.cw3.run1.TinyImage;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.Classifier;
import org.openimaj.feature.*;
import org.openimaj.image.FImage;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.annotation.Annotated;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import static org.openimaj.feature.FloatFVComparison.EUCLIDEAN;

public class LinearClassifier {

    int patchSize;
    int patchDistance;
    int clusters;
    VocabsExtractor ve;

    LinkedList<FloatFV> fv = new LinkedList<>();

    // Default Setting
    public LinearClassifier() {
        this.patchSize = 8;
        this.patchDistance = 4;
        this.clusters = 500;
    }

    public LinearClassifier(int patchSize, int patchDistance, int clusters) {
        this.patchSize = patchSize;
        this.patchDistance = patchDistance;
        this.clusters = clusters;
    }


    // compare image with the testing data, returns a guess
    public String classify(FImage fi) {
        String guess = null;

        return guess;
    }

    // create patches
    public void addTrainingData(VFSGroupDataset<FImage> data) {

        LinkedList<FloatFV> vectorList = new LinkedList<>();
        PatchExtractor pe = new PatchExtractor(this.patchSize, this.patchDistance);
        int n = 0;

        // separating the data into categories
        for (final Map.Entry<String, VFSListDataset<FImage>> entry : data.entrySet()) {
            for (FImage img : entry.getValue()) {
                vectorList.add(getVector(img));
                n += 1;
            }
        }

        System.out.println(n);
        System.out.println(vectorList.size());

        LinkedList<FloatFV> bagOfWords = new LinkedList<>();

        for (FloatFV f : vectorList) {
            bagOfWords.add(f);
        }

        LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<>(
                new VocabsExtractor(bagOfWords,patchSize,patchDistance), LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        System.out.println("Training starts!");
        ann.train(data);
        System.out.println("Training completes!");
    }

    // get vector of image
    private FloatFV getVector (FImage image) {
        FloatFV ffv;
        float[] vector = new float[image.height*image.width];

        for (int i = 0; i < image.height; i++) {
            for(int j = 0; j <image.width; j++) {
                vector[i*image.width+j] = image.pixels[i][j];
            }
        }
        ffv = new FloatFV(vector);
        return ffv;
    }

    private class PatchExtractor {

        // size of each patch
        int pSize;
        // distance between each patch
        int pDist;

        public PatchExtractor (int patchSize, int dist) {
            this.pSize = patchSize;
            this.pDist = dist;
        }

        public LinkedList<FImage> getPatches (FImage image) {
            LinkedList patches = new LinkedList<>();
                for (int i = 0; i < image.width; i += pDist){
                    for (int j = 0; j < image.height; j += pDist){
                        FImage patch = image.extractROI(i,j,pSize,pSize);
                        patch = patch.normalise();
                        patches.add(patch);
                    }
                }
            return patches;
        }

    }

    private class VocabsExtractor implements FeatureExtractor<SparseFloatFV,FImage>, FloatFVComparator {

        PatchExtractor pExtractor;
        LinkedList<FloatFV> bagOfWords;

        public VocabsExtractor (LinkedList<FloatFV> list, int ps, int pd) {
            this.pExtractor = new PatchExtractor(ps, pd);
            this.bagOfWords = list;
        }

        @Override
        public SparseFloatFV extractFeature(FImage image) {
            SparseFloatFV sffv;
            float[] wordArray = new float[bagOfWords.size()];
            LinkedList<FImage> patches = pExtractor.getPatches(image);

            for (FImage patch : patches) {
                FloatFV vector = getVector(patch);
                FloatFV closestWord = null;
                double cwDistance = Float.MAX_VALUE;

                for (FloatFV word : bagOfWords) {
                    double distance = compare(word, vector);
                    if (distance < cwDistance) {
                        closestWord = word;
                        cwDistance = distance;
                    }
                }
                // increase count
                wordArray[bagOfWords.indexOf(closestWord)] ++ ;
            }

            sffv = new SparseFloatFV(wordArray);
            return sffv;
        }

        // not using this again
        @Override
        public double compare(float[] h1, float[] h2) {
            return 0;
        }

        @Override
        public double compare(FloatFV o1, FloatFV o2) {
            System.out.println(o1);
            System.out.println(o2);
            double num = o1.compare(o2,EUCLIDEAN);
            return num;
        }

        @Override
        public boolean isDistance() {
            return false;
        }
    }

}
