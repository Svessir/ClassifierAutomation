package is.ru.machinelearning;

import java.util.Comparator;

/**
 * Object that contains the modifiable hyper parameters for a j48 tree
 */
public class J48Parameters {
    public  int minNumObj;
    public  int numFold;
    public  float confidenceFactor;

    public static final Comparator<J48Parameters> minNumObjComparator = new Comparator<J48Parameters>() {
        public int compare(J48Parameters o1, J48Parameters o2) {
            if(o1.minNumObj < o2.minNumObj)
                return -1;
            else if(o1.minNumObj > o2.minNumObj)
                return 1;
            return 0;
        }
    };

    public static final Comparator<J48Parameters> numFoldComparator = new Comparator<J48Parameters>() {
        public int compare(J48Parameters o1, J48Parameters o2) {
            if(o1.numFold < o2.numFold)
                return -1;
            else if(o1.numFold > o2.numFold)
                return 1;
            return 0;
        }
    };

    public static final Comparator<J48Parameters> confidenceFactorComparator = new Comparator<J48Parameters>() {
        public int compare(J48Parameters o1, J48Parameters o2) {
            if(o1.confidenceFactor < o2.confidenceFactor)
                return -1;
            else if(o1.confidenceFactor > o2.confidenceFactor)
                return 1;
            return 0;
        }
    };

    // error caused by this parameter setting
    public float errorPercentage;

    public J48Parameters(int minNumObj, int numFold, float confidenceFactor) {
        this.minNumObj = minNumObj;
        this.numFold = numFold;
        this.confidenceFactor = confidenceFactor;
    }

    public String toString() {
        return "minNumObj: " + minNumObj + ", numFold: " + numFold + ", confidenceFactor: " + confidenceFactor;
    }
}