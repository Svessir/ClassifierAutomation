package is.ru.machinelearning;

import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;

import java.util.Random;

/**
 * Created by Sverrir on 25.9.2016.
 */
public class AdaBoostTrainer extends AbstractTrainer {

    private AdaBoostParameters minParams;
    private AdaBoostParameters maxParams;
    private int numberOfSamples;

    public AdaBoostTrainer() {
        // Read from properties file min and max values for the hyper parameters.
        properties = loadProperties("ada-boost.properties");
        minParams = new AdaBoostParameters
                (
                        Integer.parseInt(properties.getProperty("minNumIterations")),
                        Integer.parseInt(properties.getProperty("minWeightThreshold"))
                );
        maxParams = new AdaBoostParameters
                (
                        Integer.parseInt(properties.getProperty("maxNumIterations")),
                        Integer.parseInt(properties.getProperty("maxWeightThreshold"))
                );

        numberOfSamples = Integer.parseInt(properties.getProperty("numberOfSamples"));
    }

    protected boolean continueTraining() {
        return maxParams.numIterations - minParams.numIterations >= 5 ||
                maxParams.weightThreshold - minParams.weightThreshold >= 5;
    }

    protected void loadSamples() {
        randomSamples.clear();
        Random rand = new Random();
        for(int i = 0; i < numberOfSamples; i++) {
            randomSamples.add(new AdaBoostParameters
                    (
                            rand.nextInt(maxParams.numIterations - minParams.numIterations + 1) + minParams.numIterations,
                            rand.nextInt(maxParams.weightThreshold - minParams.weightThreshold + 1) + minParams.weightThreshold
                    ));
        }
    }

    protected void newClassifier(AbstractHyperParameters params) {
        if(!(params instanceof AdaBoostParameters))
            return;

        AdaBoostM1 ada = new AdaBoostM1();
        ada.setClassifier(new J48());
        ada.setNumIterations(((AdaBoostParameters)params).numIterations);
        ada.setWeightThreshold(((AdaBoostParameters)params).weightThreshold);
        classifier = ada;
    }

    protected void updateParametersIntervals() {
        int numberOfPoints = randomSamples.size() / 4;
        int bestIndex =  getIndexOfLowestError(AdaBoostParameters.numIterationsComparator, numberOfPoints);

        minParams.numIterations = bestIndex - numberOfPoints < 0 ? ((AdaBoostParameters)randomSamples.get(0)).numIterations :
                ((AdaBoostParameters)randomSamples.get(bestIndex - numberOfPoints)).numIterations;
        maxParams.numIterations = bestIndex + numberOfPoints >= randomSamples.size() ?
                ((AdaBoostParameters)randomSamples.get(randomSamples.size() - 1)).numIterations :
                ((AdaBoostParameters)randomSamples.get(bestIndex + numberOfPoints)).numIterations;

        bestIndex = getIndexOfLowestError(AdaBoostParameters.weightThresholdComparator, numberOfPoints);

        minParams.weightThreshold = bestIndex - numberOfPoints < 0 ? ((AdaBoostParameters)randomSamples.get(0)).weightThreshold :
                ((AdaBoostParameters)randomSamples.get(bestIndex - numberOfPoints)).weightThreshold;
        maxParams.weightThreshold = bestIndex + numberOfPoints >= randomSamples.size() ?
                ((AdaBoostParameters)randomSamples.get(randomSamples.size() - 1)).weightThreshold :
                ((AdaBoostParameters)randomSamples.get(bestIndex + numberOfPoints)).weightThreshold;
    }
}
