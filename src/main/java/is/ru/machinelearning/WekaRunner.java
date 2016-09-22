package is.ru.machinelearning;

import weka.core.Instances;
import java.util.List;

/**
 * Created by Sverrir on 22.9.2016.
 */
public class WekaRunner {

    private Instances data;

    private void loadData() {

    }

    public void run() {
        // Load data from letter-recognition.arff
        loadData();

        // Get all classifier trainers that train relevant classifiers
        TrainerFactory trainerFactory = new TrainerFactory();
        List<ClassifierTrainer> trainers = trainerFactory.getTrainers();

        // Load the data to the trainers and train the classifiers
        for (ClassifierTrainer trainer: trainers) {
            trainer.setDataSet(data);
            trainer.train();
        }
    }

    public static void main(String args[]) {
        WekaRunner runner = new WekaRunner();
        runner.run();
    }
}
