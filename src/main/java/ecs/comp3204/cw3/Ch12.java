package ecs.comp3204.cw3;

import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DiskCachingFeatureExtractor;
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
import org.openimaj.image.feature.local.aggregate.PyramidSpatialAggregator;
import org.openimaj.io.IOUtils;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.IntFloatPair;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Ch12 {

    public static void main (String[] args) {

        GroupedDataset<String, VFSListDataset<Caltech101.Record<FImage>>, Caltech101.Record<FImage>> allData = null;

        try {
            allData = Caltech101.getData(ImageUtilities.FIMAGE_READER);
        } catch (IOException e) {
            e.printStackTrace();
        }

        GroupedDataset<String, ListDataset<Caltech101.Record<FImage>>, Caltech101.Record<FImage>> data =
                GroupSampler.sample(allData, 5, false); // 12.1.3 2nd parameter: allData.size()

        GroupedRandomSplitter<String, Caltech101.Record<FImage>> splits =
                new GroupedRandomSplitter<>(data, 15, 0, 15);


        DenseSIFT dsift = new DenseSIFT(5, 7);
        PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<>(dsift, 6f,  7);
        // 12.1.3
        //DenseSIFT dsift = new DenseSIFT(3, 7);
        //PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<>(dsift, 6f,  4,6,8,10);

        HardAssigner<byte[], float[], IntFloatPair> assigner =
                trainQuantiser(GroupedUniformRandomisedSampler.sample(splits.getTrainingDataset(), 30), pdsift);

        // 12.1.1
        // after using this, the accuracy of the program has increased from around 65% up to 85%
        HomogeneousKernelMap hkm = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);

        FeatureExtractor<DoubleFV, Caltech101.Record<FImage>> extractor = hkm.createWrappedExtractor(new PHOWExtractor(pdsift, assigner));

        LiblinearAnnotator<Caltech101.Record<FImage>, String> ann = new LiblinearAnnotator<>(
                extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        ann.train(splits.getTrainingDataset());

        ClassificationEvaluator<CMResult<String>, String, Caltech101.Record<FImage>> eval =
                new ClassificationEvaluator<>(
                        ann, splits.getTestDataset(), new CMAnalyser<Caltech101.Record<FImage>, String>(CMAnalyser.Strategy.SINGLE));

        Map<Caltech101.Record<FImage>, ClassificationResult<String>> guesses = eval.evaluate();


        CMResult<String> result = eval.analyse(guesses);
        System.out.println(result);

        // 12.1.2
        String filename = "savedFeature";
        File file = new File (filename);
        DiskCachingFeatureExtractor dcfe = new DiskCachingFeatureExtractor(file,extractor);

        String filename2 = "savedHardAssigner";
        File file2 = new File (filename2);

        try {
            IOUtils.writeToFile(assigner,file2);
        } catch (IOException e) {
            e.printStackTrace();
        }


    }

    static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(org.openimaj.data.dataset.Dataset<Caltech101.Record<FImage>> sample, PyramidDenseSIFT<FImage> pdsift) {
        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<>();
        for (Caltech101.Record<FImage> rec : sample) {
            FImage img = rec.getImage();
            pdsift.analyseImage(img);
            allkeys.add(pdsift.getByteKeypoints(0.005f));
        }
        if (allkeys.size() > 10000)
            allkeys = allkeys.subList(0, 10000);
        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(300);
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<>(allkeys);
        ByteCentroidsResult result = km.cluster(datasource);
        return result.defaultHardAssigner();
    }

    public static class PHOWExtractor implements FeatureExtractor<DoubleFV, Caltech101.Record<FImage>> {
        PyramidDenseSIFT<FImage> pdsift;
        HardAssigner<byte[], float[], IntFloatPair> assigner;
        public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
        {
            this.pdsift = pdsift;
            this.assigner = assigner;
        }
        public DoubleFV extractFeature(Caltech101.Record<FImage> object) {
            FImage image = object.getImage();
            pdsift.analyseImage(image);
            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner);

            BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<>(bovw, 2, 2);
            // 12.1.3
            //PyramidSpatialAggregator<byte[], SparseIntFV> spatial = new PyramidSpatialAggregator<>(bovw,2,4);
            return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
        }
    }
}
