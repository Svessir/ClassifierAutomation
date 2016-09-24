package is.ru.machinelearning;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;

/**
 * Created by Sverrir on 24.9.2016.
 */
public class TrainerOutput {
    public AbstractClassifier classifier;
    public Evaluation evaluation;

    public TrainerOutput(AbstractClassifier classifier, Evaluation evaluation) {
        this.classifier = classifier;
        this.evaluation = evaluation;
    }
}
