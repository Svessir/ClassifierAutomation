package is.ru.machinelearning;

import org.springframework.context.ApplicationContext;
import org.springframework.context.support.FileSystemXmlApplicationContext;
import java.util.List;

/**
 * Created by Sverrir on 22.9.2016.
 */
public class TrainerFactory {

    private List<ClassifierTrainer> trainers;

    public TrainerFactory() {
        load();
    }

    public List<ClassifierTrainer> getTrainers() {
        return trainers;
    }

    private void load() {
        ApplicationContext ctx = new FileSystemXmlApplicationContext("trainers.xml");
        trainers = (List<ClassifierTrainer>) ctx.getBean("trainers");
    }
}
