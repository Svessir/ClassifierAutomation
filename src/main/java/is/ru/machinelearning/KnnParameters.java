package is.ru.machinelearning;

import java.util.Comparator;

/**
 * Created by Sverrir on 28.9.2016.
 */
public class KnnParameters extends AbstractHyperParameters {
    public int KNN;
    public int windowSize;

    public static final Comparator<AbstractHyperParameters> KNNComparator = new Comparator<AbstractHyperParameters>() {
        public int compare(AbstractHyperParameters o1, AbstractHyperParameters o2) {
            if(!(o1 instanceof KnnParameters) || !(o2 instanceof KnnParameters))
                return -1;

            KnnParameters ob1 = (KnnParameters) o1;
            KnnParameters ob2 = (KnnParameters) o2;

            if(ob1.KNN < ob2.KNN)
                return -1;
            else if(ob1.KNN > ob2.KNN)
                return 1;
            return 0;
        }
    };

    public static final Comparator<AbstractHyperParameters> windowSizeComparator = new Comparator<AbstractHyperParameters>() {
        public int compare(AbstractHyperParameters o1, AbstractHyperParameters o2) {

            if(!(o1 instanceof KnnParameters) || !(o2 instanceof KnnParameters))
                return -1;

            KnnParameters ob1 = (KnnParameters) o1;
            KnnParameters ob2 = (KnnParameters) o2;

            if(ob1.windowSize < ob2.windowSize)
                return -1;
            else if(ob1.windowSize > ob2.windowSize)
                return 1;
            return 0;
        }
    };

    public KnnParameters(int KNN, int windowSize) {
        this.KNN = KNN;
        this.windowSize = windowSize;
    }
}
