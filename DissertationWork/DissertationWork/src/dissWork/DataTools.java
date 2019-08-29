/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dissWork;

import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import weka.classifiers.Classifier;
import weka.core.Debug.Random;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author yqm15fqu
 */
public class DataTools {

    public static void writeResult(Instances test, Classifier c, StringBuilder result, PrintWriter writer) throws Exception {
        result.append("No Parameter Info").append('\n');
        result.append(accuracy(c, test)).append('\n'); //The overallAccuracy Put it into a method. Build then call this.

        for (Instance instance : test) {

            double[] dist = c.distributionForInstance(instance);
            String distStr = "";
            for (int i = 0; i < dist.length; i++) {
                distStr += "," + dist[i];
            }
            result.append(c.classifyInstance(instance)).append(",").append(instance.classValue()).append(',').append(distStr).append('\n');
        }
        writer.write(result.toString());
        writer.close();
    }

    public static void writeResultEnsemble(Instances test, EnsembleBuilder c, StringBuilder result, PrintWriter writer) throws Exception {
        result.append("No Parameter Info").append('\n');
        result.append(accuracyEns(c, test)).append('\n'); //The overallAccuracy Put it into a method. Build then call this.

        double[] dist = new double[test.numClasses()];
        for (Instance instance : test) {

            double instClassVal = c.classifyEnsemble(instance);
            try {
                dist = c.distributionForInstance();
            } catch (Exception e) {
                System.out.println("You need to classify an instance before it can be written");
            }
            String distStr = "";
            for (int i = 0; i < dist.length; i++) {
                distStr += "," + dist[i];
            }
            result.append(instance.classValue()).append(",").append(instClassVal).append(',').append(distStr).append('\n');
        }
        writer.write(result.toString());
        writer.close();
    }

    public static double accuracy(Classifier c, Instances test) throws Exception {

        double result = 0;
        double accuracy = 0;
        int counter = 0;
        for (Instance instance : test) {
            result = c.classifyInstance(instance);
            if (result == instance.classValue()) {
                counter++;
            }
        }
        accuracy = (((double) counter / test.size()) * 100);

        return accuracy;
    }

    public static double accuracyEns(EnsembleBuilder c, Instances test) throws Exception {

        double result = 0;
        double accuracy = 0;
        int counter = 0;
        for (Instance instance : test) {
            result = c.classifyEnsemble(instance);
            if (result == instance.classValue()) {
                counter++;
            }
        }
        accuracy = (((double) counter / test.size()) * 100);

        return accuracy;
    }

    public static void findStats(int[] actual, int[] predicted) {

        int[] act = {1};
        int[] pre = {1};

        int[][] answer = confusionMatrix(predicted, actual);

        for (int i = 0; i < answer.length; i++) {
            for (int j = 0; j < answer.length; j++) {
                System.out.println(answer[i][j] + "\t");
            }
            System.out.println("\n");
        }
        System.out.println(Arrays.deepToString(answer));

        double a = answer[0][0];
        double b = answer[0][1];
        double c = answer[1][0];
        double d = answer[1][1];

        double TPR = a / (a + c);
        double FPR = b / (b + d);
        double FNR = c / (a + c);
        double TNR = d / (b + d);

        System.out.println("TPR: " + TPR + " FPR: " + FPR + " FNR: " + FNR + " TNR: " + TNR);
    }

    public static void runWithFolds(Classifier c, String classifName, Instances test, int folds) throws Exception {

        for (int i = 0; i < folds; i++) {
            for (int j = 0; j < test.size(); j++) {
                c.classifyInstance(test.get(j));
            }
        }
    }

    public static double measureAccuracy(Classifier c, Instances test) throws Exception {
        int counter = 0;

        for (Instance instance : test) {
            if (c.classifyInstance(instance) == instance.classValue()) {
                counter++;
            }
        }

//        System.out.println("Correctness: " + counter);
        double numInst = test.numInstances();

        System.out.println("Accuracy: " + ((double) counter / numInst) * 100);

        double accuracy = ((counter / numInst) * 100);

        return accuracy;
    }

    public static Instances loadClassificationData(String fullPath) {
        String dataLocation = fullPath;

        Instances train = null;
        try {
            FileReader reader = new FileReader(dataLocation);
            train = new Instances(reader);
        } catch (Exception e) {
            System.out.println("Exception caught: " + e);
        }
        train.setClassIndex(train.numAttributes() - 1);
        return train;
    }

    public static Instances[] splitData(Instances all, double proportion) {
        Instances[] split = new Instances[2];
        Random random = new Random();
        split[0] = new Instances(all);
        split[1] = new Instances(all, 0);
        split[0].randomize(random);

        int splitAmount = (int) Math.floor((proportion * all.numInstances()));

        for (int i = 0; i < splitAmount; i++) {
            split[1].add(split[0].remove(0));
        }
//        System.out.println(split[0].size());
//        System.out.println(split[1].size());
        return split;
    }

    public static double[] classDistribution(Instances data) {
        double[] temp = new double[data.numClasses()];
        for (int i = 0; i < data.numInstances(); i++) {
            temp[(int) data.get(i).classValue()] += 1;
        }

        for (int i = 0; i < temp.length; i++) {
            temp[i] = temp[i] / data.numInstances();
        }
        return temp;
    }

    public static int[][] confusionMatrix(int[] predicted, int[] actual) {
        int[][] confMatrix = new int[2][2];
        for (int i = 0; i < predicted.length; i++) {
            if (predicted[i] == 0) {
                if (actual[i] == predicted[i]) {
                    confMatrix[0][0] += 1;
                } else {
                    confMatrix[0][1] += 1;
                }
            } else if (actual[i] == predicted[i]) {
                confMatrix[1][1] += 1;
            } else {
                confMatrix[1][0] += 1;
            }
        }
        return confMatrix;
    }

    public static String[] listFilesForFolder(File folder) {
        File[] files = folder.listFiles();
        String[] bla = new String[files.length];
        int as = 0;
        for (final File fileEntry : folder.listFiles()) {
            if (fileEntry.isDirectory()) {
                listFilesForFolder(fileEntry);
            } else {
                bla[as] = fileEntry.getName();
                as++;
                System.out.println(fileEntry.getName());
            }
        }
        return bla;
    }

    public static int[] getClassValues(Instances data) {
        int[] classValues = new int[data.size()];
        for (int i = 0; i < data.size(); i++) {
            classValues[i] = (int) data.get(i).classValue();
        }

        return classValues;
    }

}
