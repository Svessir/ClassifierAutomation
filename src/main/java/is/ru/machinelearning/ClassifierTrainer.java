package is.ru.machinelearning;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

/**
 * Created by Sverrir on 22.9.2016.
 */
public interface ClassifierTrainer {
    TrainerOutput train();
    void setDataSet(Instances data, int numberOfTrainingInstances);
}
