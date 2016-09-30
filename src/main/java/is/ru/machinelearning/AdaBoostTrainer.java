package is.ru.machinelearning;

import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;

import java.util.ArrayList;
import java.util.Random;

/**
 * Created by Sverrir on 25.9.2016.
 */
public class AdaBoostTrainer extends AbstractTrainer {

    private AdaBoostParameters minParams;
    private AdaBoostParameters maxParams;
    private int numberOfSamples;

    private ArrayList<ArrayList<AbstractHyperParameters>> allParamsTested = new ArrayList<ArrayList<AbstractHyperParameters>>();
    private int i = 0;

    public AdaBoostTrainer() {
        // Read from properties file min and max values for the hyper parameters.
        properties = loadProperties("ada-boost.properties");
        minParams = new AdaBoostParameters
                (
                        Integer.parseInt(properties.getProperty("minNumIterations")),
                        Integer.parseInt(properties.getProperty("minWeightThreshold")),
                        Integer.parseInt(properties.getProperty("minMinNumbObj"))
                );
        maxParams = new AdaBoostParameters
                (
                        Integer.parseInt(properties.getProperty("maxNumIterations")),
                        Integer.parseInt(properties.getProperty("maxWeightThreshold")),
                        Integer.parseInt(properties.getProperty("maxMinNumbObj"))
                );

        numberOfSamples = Integer.parseInt(properties.getProperty("numberOfSamples"));
    }

    protected boolean continueTraining() {
        return  maxParams.numIterations - minParams.numIterations >= 5 ||
                maxParams.weightThreshold - minParams.weightThreshold >= 5;
    }

    protected void loadSamples() {
        randomSamples.clear();
        Random rand = new Random();
        for(int i = 0; i < numberOfSamples; i++) {
            randomSamples.add(new AdaBoostParameters
                    (
                            rand.nextInt(maxParams.numIterations - minParams.numIterations + 1) + minParams.numIterations,
                            rand.nextInt(maxParams.weightThreshold - minParams.weightThreshold + 1) + minParams.weightThreshold,
                            rand.nextInt(maxParams.minNumbObj - minParams.minNumbObj + 1) + minParams.minNumbObj
                    ));
        }
    }

    protected void newClassifier(AbstractHyperParameters params) {
        if(!(params instanceof AdaBoostParameters))
            return;

        AdaBoostM1 ada = new AdaBoostM1();
        J48 j48 = new J48();
        j48.setMinNumObj(((AdaBoostParameters) params).minNumbObj);
        ada.setClassifier(j48);
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

        bestIndex = getIndexOfLowestError(AdaBoostParameters.minNumbComparator, numberOfPoints);

        minParams.minNumbObj = bestIndex - numberOfPoints < 0 ? ((AdaBoostParameters)randomSamples.get(0)).minNumbObj :
                ((AdaBoostParameters)randomSamples.get(bestIndex - numberOfPoints)).minNumbObj;
        maxParams.minNumbObj = bestIndex + numberOfPoints >= randomSamples.size() ?
                ((AdaBoostParameters)randomSamples.get(randomSamples.size() - 1)).minNumbObj :
                ((AdaBoostParameters)randomSamples.get(bestIndex + numberOfPoints)).minNumbObj;
    }

    @Override
    protected void afterRound() {
        /**
         * Store the random samples so they can be plotted in the end of training
         */
        i++;
        if(!randomSamples.isEmpty()) {
            allParamsTested.add(new ArrayList<AbstractHyperParameters>(randomSamples));
        }
    }

    @Override
    protected void plotTrainingInfo() {
        ScatterCharter charter = new ScatterCharter();
        ArrayList<ArrayList<ScatterPoint>> numIterations = new ArrayList<ArrayList<ScatterPoint>>();
        ArrayList<ArrayList<ScatterPoint>> weightThreshold = new ArrayList<ArrayList<ScatterPoint>>();
        ArrayList<ArrayList<ScatterPoint>> minNumbObj = new ArrayList<ArrayList<ScatterPoint>>();

        for(ArrayList<AbstractHyperParameters> round : allParamsTested) {
            ArrayList<ScatterPoint> numIterationsRound = new ArrayList<ScatterPoint>();
            ArrayList<ScatterPoint> weightThresholdRound = new ArrayList<ScatterPoint>();
            ArrayList<ScatterPoint> minNumbObjRound = new ArrayList<ScatterPoint>();
            for(AbstractHyperParameters p : round) {
                AdaBoostParameters a = (AdaBoostParameters)p;
                numIterationsRound.add(new ScatterPoint(a.numIterations, a.errorPercentage));
                weightThresholdRound.add(new ScatterPoint(a.weightThreshold, a.errorPercentage));
                minNumbObjRound.add(new ScatterPoint(a.minNumbObj, a.errorPercentage));
            }
            numIterations.add(numIterationsRound);
            weightThreshold.add(weightThresholdRound);
            minNumbObj.add(minNumbObjRound);
        }

        charter.createChart(numIterations, "Number of Iterations", "Error Rate", "Number of Iterations");
        charter.createChart(weightThreshold, "Weight Threshold", "Error Rate", "Weight Threshold");
        charter.createChart(minNumbObj , "Minimum Number of Objects", "Error Rate", "AdaBoost Minimum Number of Objects");
    }
}
