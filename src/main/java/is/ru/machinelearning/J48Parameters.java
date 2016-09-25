package is.ru.machinelearning;

import java.util.Comparator;

/**
 * Object that contains the modifiable hyper parameters for a j48 tree
 */
public class J48Parameters extends AbstractHyperParameters{
    public  int minNumObj;
    public  int numFold;
    public  float confidenceFactor;

    public static final Comparator<AbstractHyperParameters> minNumObjComparator = new Comparator<AbstractHyperParameters>() {
        public int compare(AbstractHyperParameters o1, AbstractHyperParameters o2) {
            if(!(o1 instanceof J48Parameters) || !(o2 instanceof J48Parameters))
                return -1;

            J48Parameters ob1 = (J48Parameters) o1;
            J48Parameters ob2 = (J48Parameters) o2;

            if(ob1.minNumObj < ob2.minNumObj)
                return -1;
            else if(ob1.minNumObj > ob2.minNumObj)
                return 1;
            return 0;
        }
    };

    public static final Comparator<AbstractHyperParameters> numFoldComparator = new Comparator<AbstractHyperParameters>() {
        public int compare(AbstractHyperParameters o1, AbstractHyperParameters o2) {

            if(!(o1 instanceof J48Parameters) || !(o2 instanceof J48Parameters))
                return -1;

            J48Parameters ob1 = (J48Parameters) o1;
            J48Parameters ob2 = (J48Parameters) o2;

            if(ob1.numFold < ob2.numFold)
                return -1;
            else if(ob1.numFold > ob2.numFold)
                return 1;
            return 0;
        }
    };

    public static final Comparator<AbstractHyperParameters> confidenceFactorComparator = new Comparator<AbstractHyperParameters>() {
        public int compare(AbstractHyperParameters o1, AbstractHyperParameters o2) {

            if(!(o1 instanceof J48Parameters) || !(o2 instanceof J48Parameters))
                return -1;

            J48Parameters ob1 = (J48Parameters) o1;
            J48Parameters ob2 = (J48Parameters) o2;

            if(ob1.confidenceFactor < ob2.confidenceFactor)
                return -1;
            else if(ob1.confidenceFactor > ob2.confidenceFactor)
                return 1;
            return 0;
        }
    };

    public J48Parameters(int minNumObj, int numFold, float confidenceFactor) {
        this.minNumObj = minNumObj;
        this.numFold = numFold;
        this.confidenceFactor = confidenceFactor;
    }

    public String toString() {
        return "minNumObj: " + minNumObj + ", numFold: " + numFold + ", confidenceFactor: " + confidenceFactor;
    }
}