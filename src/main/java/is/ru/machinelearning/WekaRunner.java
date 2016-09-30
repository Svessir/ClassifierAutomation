package is.ru.machinelearning;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.List;

/**
 * Created by Sverrir on 22.9.2016.
 */
public class WekaRunner {

    private Instances data;
    private AbstractClassifier bestClassifier;
    private Evaluation bestEvaluation;

    private void loadData() {
        try {
            data = new Instances(
                    new BufferedReader(new FileReader("letter-recognition.arff")));
            data.setClassIndex(0);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void run() {
        // Load data from letter-recognition.arff
        loadData();

        // Get all classifier trainers that train relevant classifiers
        TrainerFactory trainerFactory = new TrainerFactory();
        List<ClassifierTrainer> trainers = trainerFactory.getTrainers();

        // Load the data to the trainers and train the classifiers
        for (ClassifierTrainer trainer: trainers) {
            trainer.setDataSet(data, 16000);
            TrainerOutput output = trainer.train();

            if(bestEvaluation == null || output.evaluation.incorrect() < bestEvaluation.incorrect()) {
                bestEvaluation = output.evaluation;
                bestClassifier = output.classifier;
            }
        }
    }

    public static void main(String args[]) {
        WekaRunner runner = new WekaRunner();
        runner.run();
    }
}
