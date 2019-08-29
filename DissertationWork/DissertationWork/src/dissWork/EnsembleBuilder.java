/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dissWork;

import java.util.ArrayList;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Rob
 */
public class EnsembleBuilder {

    private static ArrayList<Classifier> Ensemble;
    private boolean draw = false;
    private double[] EnsemblePredictions;
    private static Instances trainData;
    private static int ensembleSize = 30;
    
    public void setEnsembleSize(int x){
        this.ensembleSize = x;
    }
    

    private int findHighestPos(double[] x) {
        Random rand = new Random();
        double max = x[1];
        int value = 0;
        int doubleCounter = 0;
        double[] drawPos = new double[x.length];

        for (int i = 0; i < x.length; i++) {
            if (x[i] > max) {
                draw = false;
                max = x[i];
                drawPos = new double[x.length];
                drawPos[0] = i;
                doubleCounter = 1;
            } else if (x[i] == max) {
                draw = true;
                drawPos[doubleCounter] = i;
                doubleCounter++;
            }
        }

        if (draw) {
//            System.out.println("Draw! settled randomly."); //For debugging, remove later.
            double[] posArr = new double[doubleCounter];
            for (int i = 0; i < doubleCounter; i++) {
                posArr[i] = drawPos[i];
            }
            int chosen = rand.nextInt(posArr.length);
            value = (int) posArr[chosen];
            draw = false;
        } else {
            value = (int) drawPos[0];
        }

        return value;
    }

    //bagging with replacement
    public void buildEnsembleSingle(Classifier c, Instances train) throws Exception {
        Ensemble = new ArrayList<>(ensembleSize);
        Random rand = new Random();
        trainData = train;
        
        Instances ensembleInstances = new Instances(trainData, 0);

        int sampleSize = (int) Math.floor(trainData.size() * 0.3);

        for (int i = 0; i < ensembleSize; i++) {
            trainData.randomize(rand);
            Instances tempInst = trainData;

            for (int j = 0; j < sampleSize; j++) {
                ensembleInstances.add(tempInst.get(j));
            }
            Classifier x = c;
            x.buildClassifier(ensembleInstances);
            Ensemble.add(x);
        }

    }

    public static void buildEnsembleMulti(ArrayList<Classifier> cList, Instances train) throws Exception {
        Ensemble = new ArrayList<>(ensembleSize);
        Random rand = new Random();
        trainData = train;
        
        Instances ensembleInstances = new Instances(trainData, 0);

        int sampleSize = (int) Math.floor(trainData.size() * 0.3);

        for (int i = 0; i < ensembleSize; i++) {
            trainData.randomize(rand);
            Instances tempInst = trainData;

            for (int j = 0; j < sampleSize; j++) {
                ensembleInstances.add(tempInst.get(j));
            }
            Classifier classif = cList.get(rand.nextInt(cList.size()));
            //Builds an Ensemble of randomly picked Classifiers from the cList
            classif.buildClassifier(ensembleInstances);
            Ensemble.add(classif);
        }

    }

    public double[] getEnsemblePredictions() {
        if (EnsemblePredictions != null) {
            return EnsemblePredictions;
        } else {
            throw new NullPointerException("No instance has been classified,"
                    + " you must run classifyEnsemble() first.");
        }
    }


    public double classifyEnsemble(Instance test) throws Exception {
        EnsemblePredictions = new double[Ensemble.size()];
        for (int i = 0; i < Ensemble.size(); i++) {
            EnsemblePredictions[i] = Ensemble.get(i).classifyInstance(test);
        }

        double[] distribution = new double[test.numClasses()];

        for (int i = 0; i < Ensemble.size(); i++) {
            distribution[(int) Ensemble.get(i).classifyInstance(test)]++;
        }

        return findHighestPos(distribution);
    }

    
    public double[] distributionForInstance(){
        double[] distribution = new double[trainData.numClasses()];
        
        for (int i = 0; i < EnsemblePredictions.length; i++) {
            distribution[(int)EnsemblePredictions[i]]++;
        }
        for (int i = 0; i < distribution.length; i++) {
            distribution[i] = distribution[i]/ensembleSize;
        }
        return distribution;
    }
        
    
}
