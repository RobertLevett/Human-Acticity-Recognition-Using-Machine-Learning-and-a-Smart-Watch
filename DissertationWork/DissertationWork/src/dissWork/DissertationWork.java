/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dissWork;

import static dissWork.DataTools.*;
import static dissWork.testing.startTest;
import static dissWork.testing.startTestEnsemble;
import static dissWork.testing.startTestEnsembleMulti;
import static dissWork.testing.writeTestTime;
import java.util.ArrayList;

import multivariate_timeseriesweka.classifiers.NN_DTW_A;
import timeseriesweka.classifiers.BOSS;
import timeseriesweka.classifiers.DTW_kNN;
import timeseriesweka.classifiers.FastShapelets;
import vector_classifiers.TunedRandomForest;
import vector_classifiers.TunedRotationForest;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.OptimisedRotationForest;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Debug.Random;
import weka.core.Instances;

/**
 *
 * @author Rob
 */
public class DissertationWork {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {

        Instances data = loadClassificationData("C:\\Users\\Rob\\Desktop\\NetBeansProjects\\NetBeansProjects\\TEST\\src\\MotionData.arff");
        Random random = new Random();
        data.randomize(random);

        ArrayList<Classifier> ensembleList = new ArrayList<>();
        IBk ibk1 = new IBk();
        RandomForest ranFor1 = new RandomForest();
        RotationForest rotFor1 = new RotationForest();
        MultilayerPerceptron x21 = new MultilayerPerceptron();
        NaiveBayes nBayes1 = new NaiveBayes();
        J48 j481 = new J48();
//        BOSS boss = new BOSS();

        ensembleList.add(ibk1);
        ensembleList.add(ranFor1);
        ensembleList.add(rotFor1);
        //  ensembleList.add(x21);
        ensembleList.add(nBayes1);
        ensembleList.add(j481);
//        ensembleList.add(boss);

        System.out.println(data.firstInstance().numAttributes());

        for (int i = 0; i < 30; i++) {

            Instances[] split = splitData(data, 0.3);

            Instances train = split[0];
            Instances test = split[1];

////            TunedRandomForest x = new TunedRandomForest();
//            long startTime = System.nanoTime();
//                startTest(train, test, "TunedRandomForest", i, x, "TunedRandFor");
//            long endTime = System.nanoTime();
//            long duration = (endTime - startTime);  //divide by 1000000 to get milliseconds.
//            writeTestTime("TunedRandomForest", "TunedRanFor", duration, i);
//
//            System.out.println("TunedRandForest finished on fold: " + i);
//
//            TunedRotationForest x1 = new TunedRotationForest();
//            startTime = System.nanoTime();
//                startTest(train, test, "TunedRotationForest", i, x1, "TunedRotFor");
//            endTime = System.nanoTime();
//            duration = (endTime - startTime);
//            writeTestTime("TunedRotationForest", "TunedRotFor", duration, i);
//
//            System.out.println("TunedRotForest finished on fold: " + i);
//            OptimisedRotationForest x2 = new OptimisedRotationForest();
//            long startTime = System.nanoTime();
//                startTest(train, test, "OptimisedRotationForest", i, x2, "OptimisedRotFor");
//            long endTime = System.nanoTime();
//            long duration = (endTime - startTime);
//            writeTestTime("OptimisedRotationForest", "OptimisedRotFor", duration, i);
//
//            System.out.println("OptimisedRotForest finished on fold: " + i);
//            startTestEnsembleMulti(train, test, "RandomEnsemble", i,ensembleList, "MultiEnsemble");
//            startTestEnsemble(train,test,"Ensemble",i,rotFor1,"Ensemble");

//
//            RandomForest randFor = new RandomForest();
//            startTestEnsemble(train, test, "RandomForestEnsemble", i, randFor, "RandomForestEnsemble");
//
//            System.out.println("randomForestEnsemble fold: " + i + "done");
//
//            NaiveBayes nBayes = new NaiveBayes();
//            startTestEnsemble(train, test, "NaiveBayesEnsemble", i, nBayes, "NaiveBayesEnsemble");
//            System.out.println("NBAYES fold: " + i + "done");

            FastShapelets x = new FastShapelets();
            startTestEnsemble(train, test, "FastShapeletsEnsemble", i, x, "FastShapeletsEnsemble");
            System.out.println("FASTSHAPE fold: " + i + "done");

//           FastShapelets x = new FastShapelets(); 
//           startTest(train,test,"FastShapelets",i,x,"FastShaplets");
//            
//            RotationForest rotFor = new RotationForest();
//            startTest(train, test, "RotationForest", i, rotFor, "RotationForest");
//
//            RandomForest randFor = new RandomForest();
//            startTest(train, test, "RandomForest", i, randFor, "RandomForest");
//
//            IBk kNN = new IBk();
//            startTest(train, test, "KNN", i, kNN, "KNN");
//
//            J48 j = new J48();
//            startTest(train, test, "J48", i, j, "J48");
//
//            NaiveBayes nBayes = new NaiveBayes();
//            startTest(train, test, "NaiveBayes", i, nBayes, "NaiveBayes");
//
//            MultilayerPerceptron x2 = new MultilayerPerceptron();
//            startTest(train, test, "MultiLayerPerceptron", i, x2, "MultiLayerPercepton");
//            BOSS boss = new BOSS();
//
//            long startTime = System.nanoTime();
//            startTest(train, test, "BOSS", i, boss, "BOSS");
//            long endTime = System.nanoTime();
//            long duration = (endTime - startTime);  //divide by 1000000 to get milliseconds.

//            DTW_kNN dtwKNN = new DTW_kNN();
//            startTest(train,test,"DTWkNN",i,dtwKNN,"DTWKNN");
        }
        /*
        Okay, so. get classification accuracy of all results in all classifiers
        Do ensemble of each if time
        do one random ensemble of best classifiers
        Run through tony's stuff.
        Get critical difference diagrams
        Get diagrams of accuracies
        Confusion matrix
        Negative log likelihood
        False Positive, true positive, false negative, true negative.
        Reasoning
        Done.
        
        
         */

    }
}
