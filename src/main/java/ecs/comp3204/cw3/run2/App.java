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

public class App {

    public static void main(String [] args) throws FileSystemException {

        System.out.println("This is run2");

        // input data from url directly
        VFSGroupDataset<FImage> trainingSet = new VFSGroupDataset<>("zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip", ImageUtilities.FIMAGE_READER);
        trainingSet.remove("training");

        // Manually removing categories
        trainingSet.remove("bedroom");
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


    }

}
