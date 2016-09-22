package is.ru.machinelearning;

import weka.core.Instances;

/**
 * Created by Sverrir on 22.9.2016.
 */
public interface ClassifierTrainer {
    void train();
    void setDataSet(Instances data);
}
