package is.ru.machinelearning;
import weka.classifiers.trees.J48;

import java.util.*;

/**
 * Created by Sverrir on 22.9.2016.
 */
public class J48Trainer extends AbstractTrainer {

    private J48Parameters minParams;
    private J48Parameters maxParams;
    private int numberOfSamples;

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
     * Creates a new classifier with given hyper parameters
     *
     * @param params - Hyper parameters setting of the new classifier
     */
    protected void newClassifier(AbstractHyperParameters params) {
        if(!(params instanceof J48Parameters))
            return;

        J48 j48 = new J48();
        j48.setMinNumObj(((J48Parameters)params).minNumObj);
        j48.setNumFolds(((J48Parameters)params).numFold);
        j48.setConfidenceFactor(((J48Parameters)params).confidenceFactor);
        classifier = j48;
    }

    /**
     * Make random samples of hyper parameters in the interval of min max params
     */
    protected void loadSamples() {
        randomSamples.clear();
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

    /**
     * @return true if hyperparameter intervals are still large
     */
    protected boolean continueTraining() {
        return maxParams.minNumObj - minParams.minNumObj >= 5 ||
                maxParams.numFold - minParams.numFold >= 5 ||
                maxParams.confidenceFactor - minParams.confidenceFactor >= 0.05f;
    }

    /**
     * Calculate new hyper parameter intervals according to the error function
     */
    protected void updateParametersIntervals() {
        int numberOfPoints = randomSamples.size() / 4;
        int bestIndex =  getIndexOfLowestError(J48Parameters.minNumObjComparator, numberOfPoints);

        minParams.minNumObj = bestIndex - numberOfPoints < 0 ? ((J48Parameters)randomSamples.get(0)).minNumObj :
                ((J48Parameters)randomSamples.get(bestIndex - numberOfPoints)).minNumObj;
        maxParams.minNumObj = bestIndex + numberOfPoints >= randomSamples.size() ?
                ((J48Parameters)randomSamples.get(randomSamples.size() - 1)).minNumObj :
                ((J48Parameters)randomSamples.get(bestIndex + numberOfPoints)).minNumObj;

        bestIndex = getIndexOfLowestError(J48Parameters.numFoldComparator, numberOfPoints);

        minParams.numFold = bestIndex - numberOfPoints < 0 ? ((J48Parameters)randomSamples.get(0)).numFold :
                ((J48Parameters)randomSamples.get(bestIndex - numberOfPoints)).numFold;
        maxParams.numFold = bestIndex + numberOfPoints >= randomSamples.size() ?
                ((J48Parameters)randomSamples.get(randomSamples.size() - 1)).numFold :
                ((J48Parameters)randomSamples.get(bestIndex + numberOfPoints)).numFold;

        bestIndex = getIndexOfLowestError(J48Parameters.confidenceFactorComparator, numberOfPoints);

        minParams.confidenceFactor = bestIndex - numberOfPoints < 0 ? ((J48Parameters)randomSamples.get(0)).confidenceFactor :
                ((J48Parameters)randomSamples.get(bestIndex - numberOfPoints)).confidenceFactor;
        maxParams.confidenceFactor = bestIndex + numberOfPoints >= randomSamples.size() ?
                ((J48Parameters)randomSamples.get(randomSamples.size() - 1)).confidenceFactor :
                ((J48Parameters)randomSamples.get(bestIndex + numberOfPoints)).confidenceFactor;
    }
}
