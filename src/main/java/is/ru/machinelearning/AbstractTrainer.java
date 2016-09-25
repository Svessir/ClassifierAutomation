package is.ru.machinelearning;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Properties;
import java.util.Random;
import java.util.logging.Logger;

/**
 * Created by Sverrir on 22.9.2016.
 */
public abstract class AbstractTrainer implements ClassifierTrainer{

    protected class TrainerParameter {
        public Number value;
        public Float error;

        public TrainerParameter(Number value, Float error) {
            this.value = value;
            this.error = error;
        }
    }

    protected class ParameterInterval {
        public Number min;
        public Number max;
    }

    protected Instances dataSet;                    // Complete data
    protected int numberOfTrainingInstances;

    protected AbstractClassifier classifier;        // The classifier being trained
    protected AbstractClassifier bestClassifier;    // Best classifier found

    protected Evaluation evaluation;
    protected Evaluation bestEvaluation;

    protected Properties properties = new Properties();

    private Logger logger =
            Logger.getLogger(getClass().getName());

    public void setDataSet(Instances data, int numberOfTrainingInstances) {
        dataSet = data;
        this.numberOfTrainingInstances = numberOfTrainingInstances;
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
}
