package is.ru.machinelearning;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Sverrir on 27.9.2016.
 */
public class ScatterCharter {
    public void createChart(ArrayList<ArrayList<ScatterPoint>> points, String xName, String yName, String plotName) {
        XYSeriesCollection set = new XYSeriesCollection();
        ArrayList<XYSeries> series = new ArrayList<XYSeries>();

        int roundNumber = 0;
        for(List<ScatterPoint> list : points) {
            roundNumber++;
            XYSeries round = new XYSeries("Round " + roundNumber);
            for (ScatterPoint p : list) {
                round.add(p.x, p.y);
            }

            set.addSeries(round);
        }
        JFreeChart chart = ChartFactory.createScatterPlot(plotName, xName, yName, set, PlotOrientation.VERTICAL, true,true,true);

        File file = new File(plotName + ".png");
        try {
            ChartUtilities.saveChartAsPNG(file, chart, 800, 800);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
