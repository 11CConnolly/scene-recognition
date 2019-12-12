package ecs.comp3204.cw3.run2;

import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.annotation.Annotated;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.List;

public class App0 {

    public static void main(String [] args) throws FileSystemException {

        System.out.println("This is run2");

        // input data from url directly
        VFSGroupDataset<FImage> trainingSet = new VFSGroupDataset<>("zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip", ImageUtilities.FIMAGE_READER);
        trainingSet.remove("training");


        trainingSet.remove("bedroom");
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
        trainingSet.remove("Tallbuilding");


        System.out.println("Number of Categories provided: " + (trainingSet.size()));

//        DenseSIFT dsift = new DenseSIFT(5, 7);
//        PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<>(dsift, 6f,  7);
//
//        System.out.println("Started Creating HardAssigner");
//        HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(trainingSet,  pdsift);
//        System.out.println("Finished Creating HardAssigner");
//
//        System.out.println("Started Creating KernelMap");
//        HomogeneousKernelMap hkm = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);
//        System.out.println("Finished Creating KernelMap");
//
//        System.out.println("Started Creating Feature Extractor");
//        FeatureExtractor<DoubleFV, FImage> extractor = hkm.createWrappedExtractor(new PHOWExtractor(pdsift, assigner));
//        System.out.println("Finished Creating Feature Extractor");
//
//        LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<FImage, String>(
//                extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
//        System.out.println("Started Training");
//        ann.train(trainingSet);
//        System.out.println("Finished Training");

    }

//    static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(VFSGroupDataset<FImage> sample, PyramidDenseSIFT<FImage> pdsift) {
//        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<>();
//        int n = 0;
//        for (FImage rec : sample) {
//            FImage img = rec.getImage();
//            pdsift.analyseImage(img);
//            allkeys.add(pdsift.getByteKeypoints(0.005f));
//            System.out.println(n);
//            n += 1;
//        }
//        if (allkeys.size() > 10000)
//            allkeys = allkeys.subList(0, 10000);
//        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(300);
//        DataSource<byte[]> datasource = new LocalFeatureListDataSource<>(allkeys);
//        ByteCentroidsResult result = km.cluster(datasource);
//        return result.defaultHardAssigner();
//    }
//
//    static class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
//        PyramidDenseSIFT<FImage> pdsift;
//        HardAssigner<byte[], float[], IntFloatPair> assigner;
//        public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
//        {
//            this.pdsift = pdsift;
//            this.assigner = assigner;
//        }
//        public DoubleFV extractFeature(FImage object) {
//            FImage image = object.getImage();
//            pdsift.analyseImage(image);
//            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner);
//
//            BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<>(bovw, 2, 2);
//            return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
//        }
//    }

}
