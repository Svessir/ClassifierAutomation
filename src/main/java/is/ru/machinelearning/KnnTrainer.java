package is.ru.machinelearning;

import weka.classifiers.lazy.IBk;

import java.util.ArrayList;
import java.util.Random;

/**
 * Created by Sverrir on 28.9.2016.
 */
public class KnnTrainer extends AbstractTrainer {

    private KnnParameters minParams;
    private KnnParameters maxParams;
    private int numberOfSamples;

    private ArrayList<ArrayList<AbstractHyperParameters>> allParamsTested = new ArrayList<ArrayList<AbstractHyperParameters>>();
    private int i = 0;

    public KnnTrainer() {
        // Read from properties file min and max values for the hyper parameters.
        properties = loadProperties("knn.properties");
        minParams = new KnnParameters
                (
                        Integer.parseInt(properties.getProperty("minKNN")),
                        Integer.parseInt(properties.getProperty("minWindowSize"))
                );
        maxParams = new KnnParameters
                (
                        Integer.parseInt(properties.getProperty("maxKNN")),
                        Integer.parseInt(properties.getProperty("maxWindowSize"))
                );

        numberOfSamples = Integer.parseInt(properties.getProperty("numberOfSamples"));
    }

    protected boolean continueTraining() {
        return i < 1 && (maxParams.KNN - minParams.KNN != 0 || maxParams.windowSize - minParams.windowSize != 0);
    }

    protected void loadSamples() {
        randomSamples.clear();
        Random rand = new Random();
        for(int i = 0; i < numberOfSamples; i++) {
            randomSamples.add(new KnnParameters
                    (
                            rand.nextInt(maxParams.KNN - minParams.KNN + 1) + minParams.KNN,
                            rand.nextInt(maxParams.windowSize - minParams.windowSize + 1) + minParams.windowSize
                    ));
        }
    }

    protected void newClassifier(AbstractHyperParameters params) {
        if(!(params instanceof KnnParameters))
            return;

        IBk ibk = new IBk();
        ibk.setKNN(((KnnParameters)params).KNN);
        ibk.setWindowSize(((KnnParameters)params).windowSize);
        classifier = ibk;
    }

    protected void updateParametersIntervals() {
        int numberOfPoints = randomSamples.size() / 4;
        int bestIndex =  getIndexOfLowestError(KnnParameters.KNNComparator, numberOfPoints);

        minParams.KNN = bestIndex - numberOfPoints < 0 ? ((KnnParameters)randomSamples.get(0)).KNN :
                ((KnnParameters)randomSamples.get(bestIndex - numberOfPoints)).KNN;
        maxParams.KNN = bestIndex + numberOfPoints >= randomSamples.size() ?
                ((KnnParameters)randomSamples.get(randomSamples.size() - 1)).KNN :
                ((KnnParameters)randomSamples.get(bestIndex + numberOfPoints)).KNN;

        bestIndex = getIndexOfLowestError(KnnParameters.windowSizeComparator, numberOfPoints);

        minParams.windowSize = bestIndex - numberOfPoints < 0 ? ((KnnParameters)randomSamples.get(0)).windowSize :
                ((KnnParameters)randomSamples.get(bestIndex - numberOfPoints)).windowSize;
        maxParams.windowSize = bestIndex + numberOfPoints >= randomSamples.size() ?
                ((KnnParameters)randomSamples.get(randomSamples.size() - 1)).windowSize :
                ((KnnParameters)randomSamples.get(bestIndex + numberOfPoints)).windowSize;
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
        ArrayList<ArrayList<ScatterPoint>> KNN = new ArrayList<ArrayList<ScatterPoint>>();
        ArrayList<ArrayList<ScatterPoint>> windowSize = new ArrayList<ArrayList<ScatterPoint>>();

        for(ArrayList<AbstractHyperParameters> round : allParamsTested) {
            ArrayList<ScatterPoint> KNNRound = new ArrayList<ScatterPoint>();
            ArrayList<ScatterPoint> windowSizeRound = new ArrayList<ScatterPoint>();
            for(AbstractHyperParameters p : round) {
                KnnParameters k = (KnnParameters)p;
                KNNRound.add(new ScatterPoint(k.KNN, k.errorPercentage));
                windowSizeRound.add(new ScatterPoint(k.windowSize, k.errorPercentage));
            }
            KNN.add(KNNRound);
            windowSize.add(windowSizeRound);
        }

        charter.createChart(KNN, "KNN", "Error Rate", "KNearestNeighbours");
        charter.createChart(windowSize, "WindowSize", "Error Rate", "WindowSize");
    }
}
