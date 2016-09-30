package is.ru.machinelearning;

import weka.classifiers.functions.MultilayerPerceptron;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Random;

/**
 * Created by Sverrir on 29.9.2016.
 */
public class MultiLayerPerceptronTrainer extends AbstractTrainer {

    private MultiLayerPerceptronParameters originalMin;
    private MultiLayerPerceptronParameters originalMax;
    private MultiLayerPerceptronParameters minParams;
    private MultiLayerPerceptronParameters maxParams;
    private int numberOfSamples;
    private double learningRateBase;
    private double momentumBase;
    private int minLearningRateExponent;
    private int minMomentumExponent;
    private int maxLearningRateExponent;
    private int maxMomentumExponent;

    private ArrayList<ArrayList<AbstractHyperParameters>> allParamsTested = new ArrayList<ArrayList<AbstractHyperParameters>>();

    public MultiLayerPerceptronTrainer() {
        // Read from properties file min and max values for the hyper parameters.
        properties = loadProperties("multi-layer-perceptron.properties");
        minParams = new MultiLayerPerceptronParameters
                (
                        0,
                        0,
                        Integer.parseInt(properties.getProperty("minValidationSize")),
                        Integer.parseInt(properties.getProperty("minValidationThreshold")),
                        Integer.parseInt(properties.getProperty("minTrainingTime"))
                );
        maxParams = new MultiLayerPerceptronParameters
                (
                        1,
                        1,
                        Integer.parseInt(properties.getProperty("maxValidationSize")),
                        Integer.parseInt(properties.getProperty("maxValidationThreshold")),
                        Integer.parseInt(properties.getProperty("maxTrainingTime"))
                );

        learningRateBase = Double.parseDouble(properties.getProperty("learningRateBase"));
        momentumBase = Double.parseDouble(properties.getProperty("momentumBase"));

        minLearningRateExponent = Integer.parseInt(properties.getProperty("minLearningRateExponent"));
        maxLearningRateExponent = Integer.parseInt(properties.getProperty("maxLearningRateExponent"));

        minMomentumExponent = Integer.parseInt(properties.getProperty("minMomentumExponent"));
        maxMomentumExponent = Integer.parseInt(properties.getProperty("maxMomentumExponent"));

        originalMin = new MultiLayerPerceptronParameters(minParams.learningRate,
                minParams.momentum,
                minParams.validationSize,
                minParams.validationThreshold,
                minParams.trainingTime);

        originalMax = new MultiLayerPerceptronParameters(maxParams.learningRate,
                maxParams.momentum,
                maxParams.validationSize,
                maxParams.validationThreshold,
                maxParams.trainingTime);

        numberOfSamples = Integer.parseInt(properties.getProperty("numberOfSamples"));
    }

    protected boolean continueTraining() {
        // Terminate when all params have been narrowed down to 1% of the interval of the original params
        return  numberOfSamples > 10 &&
                (maxParams.learningRate - minParams.learningRate)/(originalMax.learningRate - originalMin.learningRate) > 0.01 ||
                (maxParams.momentum - minParams.momentum) / (originalMax.momentum - originalMin.momentum) > 0.01 ||
                (maxParams.validationSize - minParams.validationSize) / (originalMax.validationSize - minParams.validationSize) > 0.1 ||
                (maxParams.validationThreshold - minParams.validationThreshold) / (originalMax.validationThreshold - originalMin.validationThreshold) > 0.1 ||
                (maxParams.trainingTime - minParams.trainingTime) / (originalMax.trainingTime - originalMin.trainingTime) > 0.1;
    }

    protected void loadSamples() {
        System.out.println("minParams: " + minParams);
        System.out.println("maxParams: " + maxParams);
        randomSamples.clear();
        Random rand = new Random();
        for(int i = 0; i < numberOfSamples; i++) {
            randomSamples.add(new MultiLayerPerceptronParameters
                    (
                            getRandomLogScaleValue(learningRateBase, minLearningRateExponent, maxLearningRateExponent, rand),
                            getRandomLogScaleValue(momentumBase, minMomentumExponent, maxMomentumExponent, rand),
                            rand.nextInt(maxParams.validationSize - minParams.validationSize + 1) + minParams.validationSize,
                            rand.nextInt(maxParams.validationThreshold - minParams.validationThreshold + 1) + minParams.validationThreshold,
                            rand.nextInt(maxParams.trainingTime - minParams.trainingTime + 1) + minParams.trainingTime
                    )
            );
        }
    }

    protected void newClassifier(AbstractHyperParameters params) {
        if(!(params instanceof MultiLayerPerceptronParameters))
            return;

        MultilayerPerceptron mlp = new MultilayerPerceptron();
        mlp.setLearningRate(((MultiLayerPerceptronParameters) params).learningRate);
        mlp.setMomentum(((MultiLayerPerceptronParameters) params).momentum);
        mlp.setValidationSetSize(((MultiLayerPerceptronParameters) params).validationSize);
        mlp.setValidationThreshold(((MultiLayerPerceptronParameters) params).validationThreshold);
        mlp.setTrainingTime(((MultiLayerPerceptronParameters) params).trainingTime);
        mlp.setHiddenLayers("a");
        classifier = mlp;
    }

    protected void updateParametersIntervals() {
        int numberOfPoints = randomSamples.size() / 4;
        if(numberOfSamples > 10)
            numberOfSamples /= 2;
        updateLearningRate();
        updateMomentum();

        int bestIndex =  getIndexOfLowestError(MultiLayerPerceptronParameters.validationSizeComparator, numberOfPoints);

        minParams.validationSize = bestIndex - numberOfPoints < 0 ? ((MultiLayerPerceptronParameters)randomSamples.get(0)).validationSize :
                ((MultiLayerPerceptronParameters)randomSamples.get(bestIndex - numberOfPoints)).validationSize;
        maxParams.validationSize = bestIndex + numberOfPoints >= randomSamples.size() ?
                ((MultiLayerPerceptronParameters)randomSamples.get(randomSamples.size() - 1)).validationSize :
                ((MultiLayerPerceptronParameters)randomSamples.get(bestIndex + numberOfPoints)).validationSize;

        bestIndex =  getIndexOfLowestError(MultiLayerPerceptronParameters.validationThresholdComparator, numberOfPoints);

        minParams.validationThreshold = bestIndex - numberOfPoints < 0 ? ((MultiLayerPerceptronParameters)randomSamples.get(0)).validationThreshold :
                ((MultiLayerPerceptronParameters)randomSamples.get(bestIndex - numberOfPoints)).validationThreshold;
        maxParams.validationSize = bestIndex + numberOfPoints >= randomSamples.size() ?
                ((MultiLayerPerceptronParameters)randomSamples.get(randomSamples.size() - 1)).validationThreshold :
                ((MultiLayerPerceptronParameters)randomSamples.get(bestIndex + numberOfPoints)).validationThreshold;

        bestIndex =  getIndexOfLowestError(MultiLayerPerceptronParameters.trainingTimeComparator, numberOfPoints);

        minParams.trainingTime = bestIndex - numberOfPoints < 0 ? ((MultiLayerPerceptronParameters)randomSamples.get(0)).trainingTime :
                ((MultiLayerPerceptronParameters)randomSamples.get(bestIndex - numberOfPoints)).trainingTime;
        maxParams.trainingTime = bestIndex + numberOfPoints >= randomSamples.size() ?
                ((MultiLayerPerceptronParameters)randomSamples.get(randomSamples.size() - 1)).trainingTime :
                ((MultiLayerPerceptronParameters)randomSamples.get(bestIndex + numberOfPoints)).trainingTime;
    }

    private void updateLearningRate() {

        int bestLearningExponent = findLowestErrorExponent("learningRate");

        if(bestLearningExponent != minLearningRateExponent)
            minParams.learningRate = Math.pow(learningRateBase, bestLearningExponent - 1);
        if(bestLearningExponent != maxLearningRateExponent)
            maxParams.learningRate = Math.pow(learningRateBase, bestLearningExponent + 1);

        learningRateBase = bestLearningExponent != minLearningRateExponent && bestLearningExponent != maxLearningRateExponent ?
                Math.pow(Math.E, Math.log(learningRateBase)/5) : Math.pow(Math.E, Math.log(learningRateBase)/10);

        minLearningRateExponent = (int)Math.round(Math.log(minParams.learningRate) / Math.log(learningRateBase));
        maxLearningRateExponent = (int)Math.round(Math.log(maxParams.learningRate) / Math.log(learningRateBase));
    }

    private void updateMomentum() {

        int bestMomentumExponent = findLowestErrorExponent("momentum");

        if(bestMomentumExponent != minLearningRateExponent)
            minParams.momentum = Math.pow(momentumBase, bestMomentumExponent - 1);
        if(bestMomentumExponent != maxLearningRateExponent)
            maxParams.momentum = Math.pow(momentumBase, bestMomentumExponent + 1);

        momentumBase = bestMomentumExponent != minMomentumExponent && bestMomentumExponent != maxMomentumExponent ?
                Math.pow(Math.E, Math.log(momentumBase)/5) : Math.pow(Math.E, Math.log(momentumBase)/10);

        minMomentumExponent = (int)Math.round(Math.log(minParams.momentum) / Math.log(momentumBase));
        maxMomentumExponent = (int)Math.round(Math.log(maxParams.momentum) / Math.log(momentumBase));
    }

    @Override
    protected void afterRound() {
        /**
         * Store the random samples so they can be plotted in the end of training
         */
        if(!randomSamples.isEmpty()) {
            allParamsTested.add(new ArrayList<AbstractHyperParameters>(randomSamples));
        }
    }

    @Override
    protected void plotTrainingInfo() {
        ScatterCharter charter = new ScatterCharter();
        ArrayList<ArrayList<ScatterPoint>> learningRate = new ArrayList<ArrayList<ScatterPoint>>();
        ArrayList<ArrayList<ScatterPoint>> momentum = new ArrayList<ArrayList<ScatterPoint>>();
        ArrayList<ArrayList<ScatterPoint>> validationSetSize = new ArrayList<ArrayList<ScatterPoint>>();
        ArrayList<ArrayList<ScatterPoint>> validationThreshold = new ArrayList<ArrayList<ScatterPoint>>();
        ArrayList<ArrayList<ScatterPoint>> trainingTime = new ArrayList<ArrayList<ScatterPoint>>();

        for(ArrayList<AbstractHyperParameters> round : allParamsTested) {

            ArrayList<ScatterPoint> learningRateRound = new ArrayList<ScatterPoint>();
            ArrayList<ScatterPoint> momentumRound = new ArrayList<ScatterPoint>();
            ArrayList<ScatterPoint> validationSetSizeRound = new ArrayList<ScatterPoint>();
            ArrayList<ScatterPoint> validationThresholdRound = new ArrayList<ScatterPoint>();
            ArrayList<ScatterPoint> trainingTimeRound = new ArrayList<ScatterPoint>();

            for(AbstractHyperParameters p : round) {
                MultiLayerPerceptronParameters m = (MultiLayerPerceptronParameters)p;
                learningRateRound.add(new ScatterPoint(m.learningRate ,m.errorPercentage));
                momentumRound.add(new ScatterPoint(m.momentum ,m.errorPercentage));
                validationSetSizeRound.add(new ScatterPoint(m.validationSize ,m.errorPercentage));
                validationThresholdRound.add(new ScatterPoint(m.validationThreshold ,m.errorPercentage));
                trainingTimeRound.add(new ScatterPoint(m.trainingTime ,m.errorPercentage));
            }

            learningRate.add(learningRateRound);
            momentum.add(momentumRound);
            validationSetSize.add(validationSetSizeRound);
            validationThreshold.add(validationSetSizeRound);
            trainingTime.add(trainingTimeRound);
        }

        charter.createChart(learningRate, "LearningRate", "Error Rate", "Learning Rate");
        charter.createChart(momentum, "momentum", "Error Rate", "Momentum");
        charter.createChart(validationSetSize, "validationSetSize", "Error Rate", "Validation Set Size");
        charter.createChart(validationThreshold, "validationThreshold", "Error Rate", "Validation Threshold");
        charter.createChart(trainingTime, "trainingTime", "Error Rate", "Training Time");
    }

    private double getRandomLogScaleValue(double base, int minExponent, int maxExponent, Random rand) {
        int randomExponent = minExponent + (int)(rand.nextDouble() * ((maxExponent - minExponent) + 1));
        return Math.pow(base, randomExponent);
    }

    private int findLowestErrorExponent(String field) {
        boolean isLearningRate = false;

        if(field.equals("learningRate"))
            isLearningRate = true;
        else if(!field.equals("momentum"))
            return 0;

        Comparator<AbstractHyperParameters> comparator = isLearningRate ?
                MultiLayerPerceptronParameters.learningRateComparator : MultiLayerPerceptronParameters.momentumComparator;
        double base = isLearningRate ? learningRateBase : momentumBase;
        double logBase = Math.log(base);
        int minExponent = isLearningRate ? minLearningRateExponent : minMomentumExponent;
        int maxExponent = isLearningRate ? maxLearningRateExponent : maxMomentumExponent;

        System.out.println("minExponent: " + minExponent);
        System.out.println("maxExponent: " + maxExponent);
        System.out.println("logBase: " + logBase);
        System.out.println("size: " + (maxExponent - minExponent + 1));

        Collections.sort(randomSamples, comparator);

        double[][] errorList = new double[maxExponent - minExponent + 1][2];

        for(AbstractHyperParameters p : randomSamples) {
            MultiLayerPerceptronParameters m = (MultiLayerPerceptronParameters) p;
            double value = isLearningRate ? m.learningRate : m.momentum;
            int exponent = (int)Math.round(Math.log(value) / logBase);
            int index = exponent < 0 ? -exponent : exponent;
            index += maxExponent;

            System.out.println("Exponent: " + exponent);
            System.out.println("index: " + index);

            errorList[index][0] += m.errorPercentage;
            errorList[index][1] += 1.0;
        }

        double lowestError = Double.MAX_VALUE;
        int bestExponent = 0;

        for(int i = 0; i < errorList.length; i++) {

            if(errorList[i][1] == 0)
                continue;

            double avgError = errorList[i][0]/errorList[i][1];

            if(avgError < lowestError) {
                lowestError = avgError;
                bestExponent = -(i - maxExponent);
            }
        }

        System.out.println("bestExponent:" + bestExponent);
        return bestExponent;
    }
}
