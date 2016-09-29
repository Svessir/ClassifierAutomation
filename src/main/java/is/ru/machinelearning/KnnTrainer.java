package is.ru.machinelearning;

import weka.classifiers.lazy.IBk;

import java.util.ArrayList;
import java.util.Random;

/**
 * Created by Sverrir on 28.9.2016.
 */
public class KnnTrainer extends AbstractTrainer {

    private final KnnParameters originalMinParams;
    private final KnnParameters originalMaxParams;
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
                        Integer.parseInt(properties.getProperty("minKNN"))
                );
        maxParams = new KnnParameters
                (
                        Integer.parseInt(properties.getProperty("maxKNN"))
                );

        originalMinParams = new KnnParameters(minParams.KNN);
        originalMaxParams = new KnnParameters(maxParams.KNN);

        numberOfSamples = Integer.parseInt(properties.getProperty("numberOfSamples"));
    }

    protected boolean continueTraining() {
        return !(maxParams.KNN - minParams.KNN < 10);
    }

    protected void loadSamples() {
        randomSamples.clear();
        Random rand = new Random();
        for(int i = 0; i < numberOfSamples; i++) {
            randomSamples.add(new KnnParameters
                    (
                            rand.nextInt(maxParams.KNN - minParams.KNN + 1) + minParams.KNN
                    ));
        }
    }

    protected void newClassifier(AbstractHyperParameters params) {
        if(!(params instanceof KnnParameters))
            return;

        IBk ibk = new IBk();
        ibk.setKNN(((KnnParameters)params).KNN);
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

        for(ArrayList<AbstractHyperParameters> round : allParamsTested) {
            ArrayList<ScatterPoint> KNNRound = new ArrayList<ScatterPoint>();
            for(AbstractHyperParameters p : round) {
                KnnParameters k = (KnnParameters)p;
                KNNRound.add(new ScatterPoint(k.KNN, k.errorPercentage));
            }
            KNN.add(KNNRound);
        }

        charter.createChart(KNN, "KNN", "Error Rate", "K Nearest Neighbours");
    }
}
