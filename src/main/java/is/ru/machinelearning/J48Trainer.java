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

        while(continueTraining()) {
            System.out.println(minParams);
            System.out.println(maxParams);
            loadSamples();
            int i = 0;
            for(J48Parameters params : randomSamples ) {
                i++;
                newClassifier(params);

                try {
                    trainClassifier();
                    params.errorPercentage = (float) (evaluation.incorrect() / evaluation.numInstances());
                }
                catch (Exception e) {
                    e.printStackTrace();
                }
                System.out.println(i + " " + params);
            }
            updateParametersIntervals();
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
    private boolean continueTraining() {
        return maxParams.minNumObj - minParams.minNumObj >= 5 ||
                maxParams.numFold - minParams.numFold >= 5 ||
                maxParams.confidenceFactor - minParams.confidenceFactor >= 0.05f;
    }

    /**
     * Calculate new hyper parameter intervals according to the error function
     */
    private void updateParametersIntervals() {
        int numberOfPoints = randomSamples.size() / 4;
        int bestIndex =  getIndexOfLowestError(J48Parameters.minNumObjComparator, numberOfPoints);

        minParams.minNumObj = bestIndex - numberOfPoints < 0 ? randomSamples.get(0).minNumObj :
                randomSamples.get(bestIndex - numberOfPoints).minNumObj;
        maxParams.minNumObj = bestIndex + numberOfPoints >= randomSamples.size() ? randomSamples.get(randomSamples.size() - 1).minNumObj :
                randomSamples.get(bestIndex + numberOfPoints).minNumObj;

        bestIndex = getIndexOfLowestError(J48Parameters.numFoldComparator, numberOfPoints);

        minParams.numFold = bestIndex - numberOfPoints < 0 ? randomSamples.get(0).numFold :
                randomSamples.get(bestIndex - numberOfPoints).numFold;
        maxParams.numFold = bestIndex + numberOfPoints >= randomSamples.size() ? randomSamples.get(randomSamples.size() - 1).numFold :
                randomSamples.get(bestIndex + numberOfPoints).numFold;

        bestIndex = getIndexOfLowestError(J48Parameters.confidenceFactorComparator, numberOfPoints);

        minParams.confidenceFactor = bestIndex - numberOfPoints < 0 ? randomSamples.get(0).confidenceFactor :
                randomSamples.get(bestIndex - numberOfPoints).confidenceFactor;
        maxParams.confidenceFactor = bestIndex + numberOfPoints >= randomSamples.size() ? randomSamples.get(randomSamples.size() - 1).confidenceFactor :
                randomSamples.get(bestIndex + numberOfPoints).confidenceFactor;
    }

    /**
     * Get location of a estimated error minimum for a hyper parameter.
     *
     * @param comparator Comparator used to sort the samples according to a hyper parameter
     * @param numberOfPoints Number of samples used in an interval
     * @return The index of a center point in an interval of numberOfPoints samples with the lowest average error
     */
    private int getIndexOfLowestError(Comparator<J48Parameters> comparator, int numberOfPoints) {
        int bestIndex = 0;
        int halfNumberOfPoints = numberOfPoints/2;
        float currentAvg = 0;
        float bestError = Float.MAX_VALUE;

        // Initialize error array
        float[] avgErrors = new float[randomSamples.size()];
        Arrays.fill(avgErrors, Float.MAX_VALUE);

        // Sort according to hyper parameter comparator
        Collections.sort(randomSamples, comparator);

        // Get first average error interval
        for(int i = 0; i <= halfNumberOfPoints; i++)
            currentAvg += randomSamples.get(i).errorPercentage;

        currentAvg /= numberOfPoints;
        avgErrors[halfNumberOfPoints] = currentAvg;

        // Calculate the next average interval according from previous interval
        for(int i = halfNumberOfPoints + 1; i < randomSamples.size() - halfNumberOfPoints - 1 ; i++) {
            avgErrors[i] = ((avgErrors[i - 1] * numberOfPoints) - randomSamples.get((i-1) - halfNumberOfPoints).errorPercentage
                    + randomSamples.get(i + halfNumberOfPoints).errorPercentage)/numberOfPoints;
        }

        // Find lowest error interval
        for(int i = 0; i < avgErrors.length; i++) {
            if(avgErrors[i] < bestError) {
                bestError = avgErrors[i];
                bestIndex = i;
            }
        }

        return bestIndex;
    }
}
