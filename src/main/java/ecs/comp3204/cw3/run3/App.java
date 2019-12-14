package ecs.comp3204.cw3.run3;

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
import org.openimaj.image.annotation.evaluation.datasets.Caltech101;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.aggregate.PyramidSpatialAggregator;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.list.RandomisableList;
import org.openimaj.util.pair.IntFloatPair;

import java.io.*;
import java.util.*;

public class App {

    public static void main(String[] args) throws IOException {

        System.out.println("This is run #3");

        // key parameters
        final int size = 8; // 8*8
        final int step = 4; // 4 pixel apart

        // input data from url directly
        VFSGroupDataset<FImage> trainingSet = new VFSGroupDataset<>("zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip", ImageUtilities.FIMAGE_READER);
        // the training folder contains everything, do not want anything to be named training xD
        trainingSet.remove("training");

//        //Manually removing categories by commenting them out so it is easier to test
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

        DenseSIFT dsift = new DenseSIFT(4, 10);
        PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<>(dsift, 6f,  4,8);

        System.out.println("Make assigner");
        HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(trainingSet, pdsift);
        System.out.println("Finished making assigner");

        // creates a homogeneous kernel map
        HomogeneousKernelMap hkm = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);
        System.out.println("Homogeneous Kernel Map created");

        System.out.println("Make extractor");
        FeatureExtractor<DoubleFV, FImage> extractor = hkm.createWrappedExtractor(new PHOWExtractor(pdsift, assigner));
        System.out.println("Finished making extractor");

        System.out.println("Make annotator");
        LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<>(
                extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.2, 0.00001);
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

        String pathname = "run3.txt";
        File run2output = new File(pathname);
        if (!run2output.exists()) {
            run2output.createNewFile();
        }

        // writing to file "run3.txt"
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


    private static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(VFSGroupDataset<FImage> dataset, PyramidDenseSIFT<FImage> pdsift) {
        LocalFeatureList<ByteDSIFTKeypoint> allFeatures = new MemoryLocalFeatureList<>();
        for (final Map.Entry<String, VFSListDataset<FImage>> entry : dataset.entrySet()) {
            for (FImage img : entry.getValue()) {
                pdsift.analyseImage(img);
                allFeatures.addAll(pdsift.getByteKeypoints(0.005f));
            }
        }
        //System.out.println();
        System.out.println("Total number of features: " + allFeatures.size());
        if (allFeatures.size() > 10000) {
            allFeatures = allFeatures.subList(0, 10000);
        }
        // try 500 clusters
        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(500);
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<>(allFeatures);
        ByteCentroidsResult result = km.cluster(datasource);
        return result.defaultHardAssigner();
    }

    public static class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
        PyramidDenseSIFT<FImage> pdsift;
        HardAssigner<byte[], float[], IntFloatPair> assigner;
        public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
        {
            this.pdsift = pdsift;
            this.assigner = assigner;
        }

        @Override
        public DoubleFV extractFeature(FImage object) {
            FImage image = object.getImage();
            pdsift.analyseImage(image);
            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner);
            PyramidSpatialAggregator<byte[], SparseIntFV> spatial = new PyramidSpatialAggregator<>(bovw,2,4);
            return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
        }
    }
}