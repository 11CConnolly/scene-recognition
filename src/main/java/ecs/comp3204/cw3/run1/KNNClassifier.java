package ecs.comp3204.cw3.run1;

import com.fasterxml.jackson.databind.util.TypeKey;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.FloatFVComparator;
import org.openimaj.image.FImage;

import java.util.*;

import static org.openimaj.feature.FloatFVComparison.EUCLIDEAN;

public class KNNClassifier implements FloatFVComparator {

    HashMap<FloatFV, String> categories = new HashMap<>();

    // takes an image and a k value, returns the guess of which category the image belongs to
    public String classify (FImage image, int k) {

        String guess = null;
        // create a tiny image of the image provided
        TinyImage tinyImage = new TinyImage(image, k);

        FloatFV fv = tinyImage.getImageVector();
        //System.out.println(fv);
        PriorityQueue<Double> pq = new PriorityQueue<>(1);

        // sorting the list in order with the help of a priority queue
        for (FloatFV v : categories.keySet()) {
            double diff = compare(fv,v);
            pq.add(diff);
            //System.out.println("Compared To: " + categories.get(v) + " -- Difference: " + diff);
        }

        //System.out.println(Arrays.toString(pq.toArray()));

        // getting the top 30 matches out
        LinkedList<Double> ll = new LinkedList<>();
        for (int i = 0; i < k; i++ ) {
            ll.add(pq.poll());
        }

        // reverse engineer to get the FloatFV value back
        HashMap<String, Integer> votes = new HashMap<>();
        for (FloatFV v : categories.keySet()) {
            double diff = compare(fv,v);
            if (ll.contains(diff)){
                String category = categories.get(v);
                //System.out.println("1 Vote for " + category);
                if (votes.containsKey(category)) {
                    votes.put(category, (votes.get(category)+1));
                } else {
                    votes.put(category, 1);
                }
            }
        }

        //System.out.println("There are " + votes.size() + " different categories being voted!");

        int n = 0;

        for (String option : votes.keySet()){
            int num = votes.get(option);
            if (num > n) {
                n = num;
                guess = option;
            } else if (num == n) {
                guess = guess + " and " + option;
            }
            //System.out.println(option + " has " + num + " vote(s)!");
        }

        //System.out.println("The guess is "+ guess);

        return guess;
    }

    // importing training data
    public void addTrainingData(VFSGroupDataset<FImage> data, int k) {
        // separating the data into categories
        for (final Map.Entry<String, VFSListDataset<FImage>> entry : data.entrySet()) {
            for (FImage img : entry.getValue()) {
                // for each image, create a tiny image
                TinyImage ti = new TinyImage(img,k);
                // add the (vector,categoryName) to the hashmap
                categories.put(ti.getImageVector(), entry.getKey());
            }
        }
    }

    // for testing purposes
    public void displayCategories() {
        for (FloatFV fv: categories.keySet()){
            String key = fv.toString();
            String value = categories.get(fv).toString();
            System.out.println(key + " " + value);
        }
    }

    @Override
    public double compare(float[] h1, float[] h2) {
        return 0;
    }

    @Override
    public double compare(FloatFV o1, FloatFV o2) {
        double num = o1.compare(o2,EUCLIDEAN);
        return num;
    }

    @Override
    public boolean isDistance() {
        return true;
    }
}

//        -----Code from Tutorial-----
//        // Display the categories
//        for (final Map.Entry<String, VFSListDataset<FImage>> entry : trainingSet.entrySet()) {
//            DisplayUtilities.display(entry.getKey(), entry.getValue());
//            DisplayUtilities.display(entry.getKey(), entry.getValue().getInstance(new Random().nextInt(entry.getValue().size())));
//        }
//        -----------------------------

