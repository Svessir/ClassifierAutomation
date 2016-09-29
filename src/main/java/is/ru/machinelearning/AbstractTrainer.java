package is.ru.machinelearning;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;
import java.util.logging.Logger;

/**
 * Created by Sverrir on 22.9.2016.
 */
public abstract class AbstractTrainer implements ClassifierTrainer{

    protected Instances dataSet;                    // Complete data
    protected int numberOfTrainingInstances;

    protected AbstractClassifier classifier;        // The classifier being trained
    protected AbstractClassifier bestClassifier;    // Best classifier found

    protected Evaluation evaluation;
    protected Evaluation bestEvaluation;

    protected ArrayList<AbstractHyperParameters> randomSamples = new ArrayList<AbstractHyperParameters>();

    protected Properties properties = new Properties();

    private Logger logger =
            Logger.getLogger(getClass().getName());

    public void setDataSet(Instances data, int numberOfTrainingInstances) {
        dataSet = data;
        this.numberOfTrainingInstances = numberOfTrainingInstances;
    }

    public TrainerOutput train() {
        // Create and train on samples until break condition is met
        while(continueTraining()) {
            loadSamples();
            int i = 0;
            for(AbstractHyperParameters params : randomSamples ) {
                i++;
                newClassifier(params);

                try {
                    trainClassifier();
                    params.errorPercentage = (float) (evaluation.incorrect() / evaluation.numInstances());
                }
                catch (Exception e) {
                    e.printStackTrace();
                }
                System.out.println(i + " " + params + ". error: " + params.errorPercentage);
            }
            afterRound();
            updateParametersIntervals();
        }

        plotTrainingInfo();
        System.out.println(bestEvaluation.toSummaryString());
        return new TrainerOutput(bestClassifier, bestEvaluation);
    }

    protected void trainClassifier() throws Exception {
        int numberOfTestInstances = dataSet.size() - numberOfTrainingInstances;
        dataSet.randomize(new Random());
        Instances trainSet = new Instances(dataSet, 0, numberOfTrainingInstances);
        Instances testSet = new Instances(dataSet, numberOfTrainingInstances, numberOfTestInstances);

        // train classifier on complete file for tree
        classifier.buildClassifier(trainSet);

        evaluation = new Evaluation(trainSet);
        evaluation.evaluateModel(classifier, testSet);

        updateStatus();
    }

    protected Properties loadProperties(String filename) {
        try
        {
            properties.load(new FileInputStream(new File(filename)));
        }
        catch (FileNotFoundException fnfex)
        {
            String msg = "Trainer: File '" + filename + "' not found.";
            logger.severe(msg);
            fnfex.printStackTrace();
        }
        catch (IOException ioex)
        {
            String msg = "Trainer: Unable to read file '" + filename + "'.";
            logger.severe(msg);
            ioex.printStackTrace();
        }
        return properties;
    }

    protected void updateStatus() {
        if(evaluation == null)
            return;
        if(bestEvaluation == null || evaluation.incorrect() < bestEvaluation.incorrect()) {
            bestEvaluation = evaluation;
            bestClassifier = classifier;
        }
    }

    /**
     * Get location of a estimated error minimum for a hyper parameter.
     *
     * @param comparator Comparator used to sort the samples according to a hyper parameter
     * @param numberOfPoints Number of samples used in an interval
     * @return The index of a center point in an interval of numberOfPoints samples with the lowest average error
     */
    protected int getIndexOfLowestError(Comparator<AbstractHyperParameters> comparator, int numberOfPoints) {
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
        for(int i = 0; i < halfNumberOfPoints; i++)
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

    /**
     * Called after each random sample round
     */
    protected void afterRound() {
    }

    protected void plotTrainingInfo() {
    }

    protected abstract boolean continueTraining();
    protected abstract void loadSamples();
    protected abstract void newClassifier(AbstractHyperParameters params);
    protected abstract void updateParametersIntervals();
}
