package is.ru.machinelearning;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Random;

/**
 * Created by Sverrir on 22.9.2016.
 */
public class J48Trainer extends AbstractTrainer {

    /**
     * Object that contains the hyper parameters for a j48 tree
     */
    private class J48Parameters {
        public final int minNumObj;
        public final int numFold;
        public final float confidenceFactor;

        public J48Parameters(int minNumObj, int numFold, float confidenceFactor) {
            this.minNumObj = minNumObj;
            this.numFold = numFold;
            this.confidenceFactor = confidenceFactor;
        }
    }

    private J48Parameters minParams;
    private J48Parameters maxParams;
    private int numberOfSamples;
    private ArrayList<J48Parameters> randomSamples = new ArrayList<J48Parameters>();

    public J48Trainer() {
        // Read from properties file min and max values for the hyper parameters.
        properties = loadProperties("j48.properties");
        minParams = new J48Parameters
                    (
                        Integer.parseInt(properties.getProperty("minMinNumObj")),
                        Integer.parseInt(properties.getProperty("minNumFold")),
                        Float.parseFloat(properties.getProperty("minConfidenceFactor"))
                    );
        maxParams = new J48Parameters
                    (
                        Integer.parseInt(properties.getProperty("maxMinNumObj")),
                        Integer.parseInt(properties.getProperty("maxNumFold")),
                        Float.parseFloat(properties.getProperty("maxConfidenceFactor"))
                    );

        numberOfSamples = Integer.parseInt(properties.getProperty("numberOfSamples"));
    }

    /**
     * Trains J48 classifiers on the data set
     *
     * @return the best j48 classifer trained
     */
    public TrainerOutput train() {
        loadSamples();
        for(J48Parameters params : randomSamples ) {
            newClassifier(params);

            try {
                trainClassifier();
            }
            catch (Exception e) {
                e.printStackTrace();
            }
        }
        System.out.println(bestEvaluation.toSummaryString());
        return new TrainerOutput(bestClassifier, bestEvaluation);
    }

    /**
     * Creates a new classifier with given hyper parameters
     *
     * @param params - Hyper parameters setting of the new classifier
     */
    private void newClassifier(J48Parameters params) {
        J48 j48 = new J48();
        j48.setMinNumObj(params.minNumObj);
        j48.setNumFolds(params.numFold);
        j48.setConfidenceFactor(params.confidenceFactor);
        classifier = j48;
    }

    private void loadSamples() {
        Random rand = new Random();
        for(int i = 0; i < numberOfSamples; i++) {
            randomSamples.add(new J48Parameters
                    (
                            rand.nextInt(maxParams.minNumObj - minParams.minNumObj + 1) + minParams.minNumObj,
                            rand.nextInt(maxParams.numFold - minParams.numFold + 1) + minParams.numFold,
                            rand.nextFloat() * (maxParams.confidenceFactor - minParams.confidenceFactor) + minParams.confidenceFactor
                    ));
        }
    }
}
