package ecs.comp3204.cw3.run2;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.*;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.list.RandomisableList;
import org.openimaj.util.pair.IntFloatPair;

import java.io.DataOutput;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

public class App {

    public static void main(String[] args) throws FileSystemException {

        System.out.println("This is run2");

        // key parameters
        final int size = 8; // 8*8
        final int step = 4; // 4 pixel apart

        // input data from url directly
        VFSGroupDataset<FImage> trainingSet = new VFSGroupDataset<>("zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip", ImageUtilities.FIMAGE_READER);
        // the training folder contains everything, do not want anything to be named training xD
        trainingSet.remove("training");

        // Manually removing categories by commenting them out so it is easy to test
        //trainingSet.remove("bedroom");
        trainingSet.remove("Coast");
        trainingSet.remove("Forest");
        trainingSet.remove("Highway");
        trainingSet.remove("industrial");
        trainingSet.remove("Insidecity");
        trainingSet.remove("kitchen");
        trainingSet.remove("livingroom");
        trainingSet.remove("Mountain");
        trainingSet.remove("Office");
        trainingSet.remove("OpenCountry");
        trainingSet.remove("store");
        trainingSet.remove("Street");
        trainingSet.remove("Suburb");
        trainingSet.remove("TallBuilding");

        System.out.println("Number of Categories provided: " + (trainingSet.size()));

        PatchExtractor pe = new PatchExtractor(size, step);

        System.out.println("make assigner");
        HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(trainingSet, pe);
        System.out.println("finished making assigner");


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
            LocalFeatureList<FloatKeypoint> floatKPList = new LocalFeatureList<FloatKeypoint>() {
                @Override
                public <Q> Q[] asDataArray(Q[] a) {
                    return null;
                }

                @Override
                public int vecLength() {
                    return 0;
                }

                @Override
                public LocalFeatureList<FloatKeypoint> subList(int fromIndex, int toIndex) {
                    return null;
                }

                @Override
                public RandomisableList<FloatKeypoint> randomSubList(int nelem) {
                    return null;
                }

                @Override
                public int size() {
                    return 0;
                }

                @Override
                public boolean isEmpty() {
                    return false;
                }

                @Override
                public boolean contains(Object o) {
                    return false;
                }

                @Override
                public Iterator<FloatKeypoint> iterator() {
                    return null;
                }

                @Override
                public Object[] toArray() {
                    return new Object[0];
                }

                @Override
                public <T> T[] toArray(T[] a) {
                    return null;
                }

                @Override
                public boolean add(FloatKeypoint floatKeypoint) {
                    return false;
                }

                @Override
                public boolean remove(Object o) {
                    return false;
                }

                @Override
                public boolean containsAll(Collection<?> c) {
                    return false;
                }

                @Override
                public boolean addAll(Collection<? extends FloatKeypoint> c) {
                    return false;
                }

                @Override
                public boolean addAll(int index, Collection<? extends FloatKeypoint> c) {
                    return false;
                }

                @Override
                public boolean removeAll(Collection<?> c) {
                    return false;
                }

                @Override
                public boolean retainAll(Collection<?> c) {
                    return false;
                }

                @Override
                public void clear() {

                }

                @Override
                public FloatKeypoint get(int index) {
                    return null;
                }

                @Override
                public FloatKeypoint set(int index, FloatKeypoint element) {
                    return null;
                }

                @Override
                public void add(int index, FloatKeypoint element) {

                }

                @Override
                public FloatKeypoint remove(int index) {
                    return null;
                }

                @Override
                public int indexOf(Object o) {
                    return 0;
                }

                @Override
                public int lastIndexOf(Object o) {
                    return 0;
                }

                @Override
                public ListIterator<FloatKeypoint> listIterator() {
                    return null;
                }

                @Override
                public ListIterator<FloatKeypoint> listIterator(int index) {
                    return null;
                }

                @Override
                public void writeASCII(PrintWriter out) throws IOException {

                }

                @Override
                public String asciiHeader() {
                    return null;
                }

                @Override
                public void writeBinary(DataOutput out) throws IOException {

                }

                @Override
                public byte[] binaryHeader() {
                    return new byte[0];
                }
            };
            LinkedList<FImage> patches = getPatches(img);
            for (FImage patch : patches) {
                FloatKeypoint fkp = new FloatKeypoint(0, 0, 0, 0, getVector(patch));
                floatKPList.add(fkp);
            }
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
        List<LocalFeatureList<FloatKeypoint>> allVectors = new ArrayList<LocalFeatureList<FloatKeypoint>>();
        for (final Map.Entry<String, VFSListDataset<FImage>> entry : dataset.entrySet()) {
            for (FImage img : entry.getValue()) {
                allVectors.add(p.analyseImage(img));
            }
        }
        System.out.println(allVectors.size());
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
            BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>(assigner);

            sffv = bovw.aggregate(pe.analyseImage(object));

            return sffv;
        }
    }
}