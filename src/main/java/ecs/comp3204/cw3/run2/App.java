package ecs.comp3204.cw3.run2;

import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.*;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.FileLocalFeatureList;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.feature.local.list.StreamLocalFeatureList;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.list.RandomisableList;
import org.openimaj.util.pair.IntFloatPair;

import java.io.*;
import java.util.*;

public class App {

    public static void main(String[] args) throws IOException {

        System.out.println("This is run2");

        // key parameters
        final int size = 8; // 8*8
        final int step = 4; // 4 pixel apart

        // input data from url directly
        VFSGroupDataset<FImage> trainingSet = new VFSGroupDataset<>("zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip", ImageUtilities.FIMAGE_READER);
        // the training folder contains everything, do not want anything to be named training xD
        trainingSet.remove("training");

        //Manually removing categories by commenting them out so it is easier to test
//        trainingSet.remove("bedroom");
//        trainingSet.remove("Coast");
//        trainingSet.remove("Forest");
//        trainingSet.remove("Highway");
//        trainingSet.remove("industrial");
//        trainingSet.remove("Insidecity");
//        trainingSet.remove("kitchen");
//        trainingSet.remove("livingroom");
//        trainingSet.remove("Mountain");
//        trainingSet.remove("Office");
//        trainingSet.remove("OpenCountry");
//        trainingSet.remove("store");
//        trainingSet.remove("Street");
//        trainingSet.remove("Suburb");
//        trainingSet.remove("TallBuilding");

        System.out.println("Number of Categories provided: " + (trainingSet.size()));

        PatchExtractor pe = new PatchExtractor(size, step);

        System.out.println("Make assigner");
        HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(trainingSet, pe);
        System.out.println("Finished making assigner");

        System.out.println("Make extractor");
        FeatureExtractor<SparseIntFV, FImage> extractor = new BOVWExtractor(pe, assigner);
        System.out.println("Finished making extractor");

        System.out.println("Make annotator");
        LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<>(
                extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        System.out.println("Finished making annotator");
        System.out.println("Start training");
        ann.train(trainingSet);
        System.out.println("Finished training");

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
        File run2output = new File(pathname);
        if (!run2output.exists()) {
            run2output.createNewFile();
        }

        // writing to file "run2.txt"
        BufferedWriter bw = new BufferedWriter(new FileWriter(run2output));

        for (File file : files) {
            if (file.isFile()) {
                FImage fi = ImageUtilities.readF(file);
                List<ScoredAnnotation<String>> list = ann.annotate(fi);
                String guess = list.toString();
                int finalChar = guess.indexOf(",");
                guess = guess.substring(2,finalChar);

                String output = file.getName()+" "+guess+"\n";
                System.out.println(output);
                bw.write(output);
            }
        }
        System.out.println("Successfully written output to file " + pathname + "!");
        bw.close();
    }

    // the machine to extract data
    private static class PatchExtractor {

        // size of each patch
        int pSize;
        // distance between each patch
        int pDist;

        public PatchExtractor(int patchSize, int dist) {
            this.pSize = patchSize;
            this.pDist = dist;
        }

        public LinkedList<FImage> getPatches(FImage image) {
            LinkedList<FImage> patches = new LinkedList<>();
            for (int i = 0; i < image.width; i += pDist) {
                for (int j = 0; j < image.height; j += pDist) {
                    FImage patch = image.extractROI(i, j, pSize, pSize);
                    patch = patch.normalise();
                    patches.add(patch);
                }
            }
            return patches;
        }

        public LocalFeatureList<FloatKeypoint> analyseImage(FImage img) {
            LocalFeatureList<FloatKeypoint> floatKPList = new MemoryLocalFeatureList<>();
            LinkedList<FImage> patches = getPatches(img);
            //System.out.println("Patches: " + patches.size());
            for (FImage patch : patches) {
                FloatKeypoint fkp = new FloatKeypoint(0, 0, 0, 0, getVector(patch));
                floatKPList.add(fkp);
            }
            //System.out.println("FloatKeyPoint List: " + floatKPList.size()); // this return 0
            return floatKPList;
        }

        // get FloatFV of image
        public FloatFV getFloatFV(FImage image) {
            FloatFV ffv;
            float[] vector = new float[image.height * image.width];

            for (int i = 0; i < image.height; i++) {
                for (int j = 0; j < image.width; j++) {
                    vector[i * image.width + j] = image.pixels[i][j];
                }
            }
            ffv = new FloatFV(vector);
            return ffv;
        }

        // get vector of image
        public float[] getVector(FImage image) {
            float[] vector = new float[image.height * image.width];

            for (int i = 0; i < image.height; i++) {
                for (int j = 0; j < image.width; j++) {
                    vector[i * image.width + j] = image.pixels[i][j];
                }
            }
            return vector;
        }

    }

    private static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(VFSGroupDataset<FImage> dataset, PatchExtractor p) {
        LocalFeatureList<FloatKeypoint> allVectors = new MemoryLocalFeatureList<>();
        for (final Map.Entry<String, VFSListDataset<FImage>> entry : dataset.entrySet()) {
            for (FImage img : entry.getValue()) {
                allVectors.addAll(p.analyseImage(img));
            }
        }
        //System.out.println();
        System.out.println("Total vector size: " + allVectors.size());
        if (allVectors.size() > 10000) {
            allVectors = allVectors.subList(0, 10000);
        }
        // try 500 clusters
        FloatKMeans km = FloatKMeans.createKDTreeEnsemble(500);
        DataSource<float[]> datasource = new LocalFeatureListDataSource<>(allVectors);
        FloatCentroidsResult result = km.cluster(datasource);
        return result.defaultHardAssigner();
    }

    static class BOVWExtractor implements FeatureExtractor<SparseIntFV, FImage> {

        PatchExtractor pe;
        HardAssigner<float[], float[], IntFloatPair> assigner;

        public BOVWExtractor(PatchExtractor pe, HardAssigner<float[], float[], IntFloatPair> assigner) {
            this.pe = pe;
            this.assigner = assigner;
        }

        @Override
        public SparseIntFV extractFeature(FImage object) {

            SparseIntFV sffv;
            BagOfVisualWords<float[]> bovw = new BagOfVisualWords<>(assigner);

            sffv = bovw.aggregate(pe.analyseImage(object));

            return sffv;
        }
    }
}