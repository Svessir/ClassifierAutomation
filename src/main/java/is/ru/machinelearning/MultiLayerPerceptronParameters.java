package is.ru.machinelearning;

import java.util.Comparator;

/**
 * Created by Sverrir on 29.9.2016.
 */
public class MultiLayerPerceptronParameters extends AbstractHyperParameters {
    public double learningRate;
    public double momentum;
    public int validationSize;
    public int validationThreshold;
    public int trainingTime;

    public static final Comparator<AbstractHyperParameters> learningRateComparator = new Comparator<AbstractHyperParameters>() {
        public int compare(AbstractHyperParameters o1, AbstractHyperParameters o2) {
            if(!(o1 instanceof MultiLayerPerceptronParameters) || !(o2 instanceof MultiLayerPerceptronParameters))
                return -1;

            MultiLayerPerceptronParameters ob1 = (MultiLayerPerceptronParameters) o1;
            MultiLayerPerceptronParameters ob2 = (MultiLayerPerceptronParameters) o2;

            if(ob1.learningRate < ob2.learningRate)
                return -1;
            else if(ob1.learningRate > ob2.learningRate)
                return 1;
            return 0;
        }
    };

    public static final Comparator<AbstractHyperParameters> momentumComparator = new Comparator<AbstractHyperParameters>() {
        public int compare(AbstractHyperParameters o1, AbstractHyperParameters o2) {
            if(!(o1 instanceof MultiLayerPerceptronParameters) || !(o2 instanceof MultiLayerPerceptronParameters))
                return -1;

            MultiLayerPerceptronParameters ob1 = (MultiLayerPerceptronParameters) o1;
            MultiLayerPerceptronParameters ob2 = (MultiLayerPerceptronParameters) o2;

            if(ob1.momentum < ob2.momentum)
                return -1;
            else if(ob1.momentum > ob2.momentum)
                return 1;
            return 0;
        }
    };

    public static final Comparator<AbstractHyperParameters> validationSizeComparator = new Comparator<AbstractHyperParameters>() {
        public int compare(AbstractHyperParameters o1, AbstractHyperParameters o2) {
            if(!(o1 instanceof MultiLayerPerceptronParameters) || !(o2 instanceof MultiLayerPerceptronParameters))
                return -1;

            MultiLayerPerceptronParameters ob1 = (MultiLayerPerceptronParameters) o1;
            MultiLayerPerceptronParameters ob2 = (MultiLayerPerceptronParameters) o2;

            if(ob1.validationSize < ob2.validationSize)
                return -1;
            else if(ob1.validationSize > ob2.validationSize)
                return 1;
            return 0;
        }
    };

    public static final Comparator<AbstractHyperParameters> trainingTimeComparator = new Comparator<AbstractHyperParameters>() {
        public int compare(AbstractHyperParameters o1, AbstractHyperParameters o2) {
            if(!(o1 instanceof MultiLayerPerceptronParameters) || !(o2 instanceof MultiLayerPerceptronParameters))
                return -1;

            MultiLayerPerceptronParameters ob1 = (MultiLayerPerceptronParameters) o1;
            MultiLayerPerceptronParameters ob2 = (MultiLayerPerceptronParameters) o2;

            if(ob1.trainingTime < ob2.trainingTime)
                return -1;
            else if(ob1.trainingTime > ob2.trainingTime)
                return 1;
            return 0;
        }
    };

    public static final Comparator<AbstractHyperParameters> validationThresholdComparator = new Comparator<AbstractHyperParameters>() {
        public int compare(AbstractHyperParameters o1, AbstractHyperParameters o2) {
            if(!(o1 instanceof MultiLayerPerceptronParameters) || !(o2 instanceof MultiLayerPerceptronParameters))
                return -1;

            MultiLayerPerceptronParameters ob1 = (MultiLayerPerceptronParameters) o1;
            MultiLayerPerceptronParameters ob2 = (MultiLayerPerceptronParameters) o2;

            if(ob1.validationThreshold < ob2.validationThreshold)
                return -1;
            else if(ob1.validationThreshold > ob2.validationThreshold)
                return 1;
            return 0;
        }
    };

    public MultiLayerPerceptronParameters(double learningRate, double momentum, int validationSize,
                                          int validationThreshold, int trainingTime) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.validationSize = validationSize;
        this.validationThreshold = validationThreshold;
        this.trainingTime = trainingTime;
    }

    public String toString() {
        return "learning Rate: " + learningRate + ", momentum: " + momentum + ", validationSetSize: " + validationSize
                + ",validationThreshold: " + validationThreshold + ", trainingTime:" + trainingTime;
    }
}
