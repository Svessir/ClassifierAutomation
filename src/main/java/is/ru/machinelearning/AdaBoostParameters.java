package is.ru.machinelearning;

import java.util.Comparator;

/**
 * Created by Sverrir on 25.9.2016.
 */
public class AdaBoostParameters extends AbstractHyperParameters{
    public int numIterations;
    public int weightThreshold;

    public AdaBoostParameters(int numIterations, int weightThreshold) {
        this.numIterations = numIterations;
        this.weightThreshold = weightThreshold;
    }

    public static final Comparator<AbstractHyperParameters> numIterationsComparator = new Comparator<AbstractHyperParameters>() {
        public int compare(AbstractHyperParameters o1, AbstractHyperParameters o2) {
            if(!(o1 instanceof AdaBoostParameters) || !(o2 instanceof AdaBoostParameters))
                return -1;

            AdaBoostParameters ob1 = (AdaBoostParameters) o1;
            AdaBoostParameters ob2 = (AdaBoostParameters) o2;

            if(ob1.numIterations < ob2.numIterations)
                return -1;
            else if(ob1.numIterations > ob2.numIterations)
                return 1;
            return 0;
        }
    };

    public static final Comparator<AbstractHyperParameters> weightThresholdComparator = new Comparator<AbstractHyperParameters>() {
        public int compare(AbstractHyperParameters o1, AbstractHyperParameters o2) {
            if(!(o1 instanceof AdaBoostParameters) || !(o2 instanceof AdaBoostParameters))
                return -1;

            AdaBoostParameters ob1 = (AdaBoostParameters) o1;
            AdaBoostParameters ob2 = (AdaBoostParameters) o2;

            if(ob1.weightThreshold < ob2.weightThreshold)
                return -1;
            else if(ob1.weightThreshold > ob2.weightThreshold)
                return 1;
            return 0;
        }
    };

    public String toString() {
        return "numIterations: " + numIterations + ", weightThreshold: " + weightThreshold;
    }
}
