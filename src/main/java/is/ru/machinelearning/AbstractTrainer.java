package is.ru.machinelearning;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

/**
 * Created by Sverrir on 22.9.2016.
 */
public abstract class AbstractTrainer implements ClassifierTrainer{

    protected Instances dataSet;                // Complete data
    protected Instances trainingData;           // training part of the data
    protected Instances testData;               // test part of the data

    protected AbstractClassifier classifier;    // The classifier used by the trainer

    public void printName() {
        System.out.println(getClass().getName());
    }
}
