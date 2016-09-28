package is.ru.machinelearning;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.List;
import java.util.Random;

/**
 * Created by Sverrir on 22.9.2016.
 */
public class WekaRunner {

    private Instances data;

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
        }
    }

    public static void main(String args[]) {
        WekaRunner runner = new WekaRunner();
        runner.run();
    }
}
