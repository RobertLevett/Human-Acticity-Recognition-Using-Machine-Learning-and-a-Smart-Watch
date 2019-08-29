/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dissWork;

import static dissWork.DataTools.writeResult;
import static dissWork.DataTools.writeResultEnsemble;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import weka.classifiers.Classifier;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.PropertyPath.Path;

/**
 *
 * @author yqm15fqu
 */
public class testing {

    public static void writeTestTime(String classifierName, String probName, long timeToExecute, int foldNo) {
        String filePath = " C:\\Users\\Rob\\Desktop\\classifierTiming";

        StringBuilder result = new StringBuilder();

        if (!new File(filePath + classifierName).exists()) {
            new File(filePath + classifierName).mkdirs();
        }

        if (!new File(filePath + classifierName + "\\" + probName + "").exists()) {
            new File(filePath + classifierName + "\\" + probName + "").mkdirs();
        }

        try (PrintWriter writer = new PrintWriter(new File(filePath + "\\" + classifierName + "\\" + probName + "\\testFold" + foldNo + ".csv"))) {
            result.append(probName).append(", ").append(classifierName).append(", ").append(timeToExecute).append(",").append(foldNo).append("\n");
            writer.write(result.toString());
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void startTest(Instances train, Instances test, String probName, int foldNo, Classifier c, String classifierName) throws FileNotFoundException, Exception {
        String filePath = "C:\\Users\\Rob\\Desktop\\dissResults2\\";
//        String x = probName.substring(0, probName.length() - 5);

        StringBuilder result = new StringBuilder();

        if (!new File(filePath + classifierName).exists()) {
            new File(filePath + classifierName).mkdirs();
        }

        if (!new File(filePath + classifierName + "\\" + probName + "").exists()) {
            new File(filePath + classifierName + "\\" + probName + "").mkdirs();
        }

        try (PrintWriter writer = new PrintWriter(new File(filePath + "\\" + classifierName + "\\" + probName + "\\testFold" + foldNo + ".csv"))) {
            c.buildClassifier(train);
            result.append(probName).append(", " + classifierName).append('\n');
            writeResult(test, c, result, writer);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void startTestEnsemble(Instances train, Instances test, String probName, int foldNo, Classifier ens, String ensembleName) throws FileNotFoundException, Exception {
        String filePath = "C:\\Users\\Rob\\Desktop\\dissResults3\\";
//        String x = probName.substring(0, probName.length() - 5);

        StringBuilder result = new StringBuilder();

        if (!new File(filePath + ensembleName).exists()) {
            new File(filePath + ensembleName).mkdirs();
        }

        if (!new File(filePath + ensembleName + "\\" + probName + "").exists()) {
            new File(filePath + ensembleName + "\\" + probName + "").mkdirs();
        }

        try (PrintWriter writer = new PrintWriter(new File(filePath + "\\" + ensembleName + "\\" + probName + "\\testFold" + foldNo + ".csv"))) {
            EnsembleBuilder ensemble = new EnsembleBuilder();
            ensemble.buildEnsembleSingle(ens, train);
            result.append(probName).append(", " + ensembleName).append('\n');
            writeResultEnsemble(test, ensemble, result, writer);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void startTestEnsembleMulti(Instances train, Instances test, String probName, int foldNo, ArrayList<Classifier> ens, String ensembleName) throws FileNotFoundException, Exception {
        String filePath = "C:\\Users\\Rob\\Desktop\\dissResults2\\";

        StringBuilder result = new StringBuilder();

        if (!new File(filePath + ensembleName).exists()) {
            new File(filePath + ensembleName).mkdirs();
        }

        if (!new File(filePath + ensembleName + "\\" + probName + "").exists()) {
            new File(filePath + ensembleName + "\\" + probName + "").mkdirs();
        }

        try (PrintWriter writer = new PrintWriter(new File(filePath + "\\" + ensembleName + "\\" + probName + "\\testFold" + foldNo + ".csv"))) {
            EnsembleBuilder ensemble = new EnsembleBuilder();
            ensemble.buildEnsembleMulti(ens, train);
            result.append(probName).append(", " + ensembleName).append('\n');
            writeResultEnsemble(test, ensemble, result, writer);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
