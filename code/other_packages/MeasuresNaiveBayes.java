package other_packages;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

public class MeasuresNaiveBayes
{
	Map<String, Integer> attributeCountP1, attributeCountN1;
	Map<String, Integer> attributeCountVals;
	public int classCountN1, classCountP1;
	double truePositive, falseNegative, falsePositive, trueNegative;
	String trainFile, testFile;
	
	public MeasuresNaiveBayes(String f1, String f2)
	{
		trainFile = f1;
		testFile = f2;
	}
	
	private void trainData() throws IOException
	{
		BufferedReader br = new BufferedReader(new FileReader(trainFile));
		String str, cls, attribute;
		String[] strList;
		
		while ((str = br.readLine()) != null) {
			strList = str.split(" ");
			cls = strList[0];
			
			if (cls.equals("-1")) {
				classCountN1++;				
			} else {
				classCountP1++;				
			}
			
			for (int i=1; i<strList.length; i++) {
				attribute = strList[i];
				
				if (cls.equals("-1")) {
					if (!attributeCountN1.containsKey(attribute)) {
						attributeCountN1.put(attribute, 0);
					}
					attributeCountN1.put(attribute, attributeCountN1.get(attribute) + 1);
				} else {
					if (!attributeCountP1.containsKey(attribute)) {
						attributeCountP1.put(attribute, 0);
					}
					attributeCountP1.put(attribute, attributeCountP1.get(attribute) + 1);
				}
				
				if (!attributeCountVals.containsKey(attribute.split(":")[0]) || 
					Integer.parseInt(attribute.split(":")[1]) > attributeCountVals.get(attribute.split(":")[0]))
				{
					attributeCountVals.put(attribute.split(":")[0], Integer.parseInt(attribute.split(":")[1]));
				}
			}
		}
		
		for (Entry<String, Integer> e : attributeCountVals.entrySet()) {
			int p = 0, n = 0;
			
			for (int i=1; i<=e.getValue(); i++) {
				if (!attributeCountP1.containsKey(e.getKey() + ":" + i)) {
					attributeCountP1.put(e.getKey() + ":" + i, 0);
				}
				if (!attributeCountN1.containsKey(e.getKey() + ":" + i)) {
					attributeCountN1.put(e.getKey() + ":" + i, 0);
				}
				
				p += attributeCountP1.get(e.getKey() + ":"  + i);
				n += attributeCountN1.get(e.getKey() + ":"  + i);
			}
			
			attributeCountP1.put(e.getKey() + ":0", classCountP1 - p);
			attributeCountN1.put(e.getKey() + ":0", classCountN1 - n);
		}
		
		br.close();
	}
	
	private void testData(String fileName) throws IOException
	{
		BufferedReader br = new BufferedReader(new FileReader(fileName));
		String str, cls, attribute, predictedClass;
		String[] strList;
		double posProbability, negProbability;
		int posCount, negCount;
		
		truePositive = 0;
		falseNegative = 0;
		falsePositive = 0;
		trueNegative = 0;
		
		while ((str = br.readLine()) != null) {
			posProbability = 1;
			negProbability = 1;
			str = str.trim();

			for (Entry<String, Integer> e : attributeCountVals.entrySet()) {
				if (!str.contains(" " + e.getKey() + ":")) {
					attribute = e.getKey() + ":0";
					
					if (!attributeCountP1.containsKey(attribute)) {
						continue;					
					}

					posCount = 1 + attributeCountP1.get(attribute);
					negCount = 1 + attributeCountN1.get(attribute);

					posProbability += Math.log(posCount/ (double) (classCountP1 + attributeCountVals.get(attribute.split(":")[0]) + 1));
					negProbability += Math.log(negCount/ (double) (classCountN1 + attributeCountVals.get(attribute.split(":")[0]) + 1));
				}
			}
			
			strList = str.trim().split(" ");
			cls = strList[0];	
			for (int i=1; i<strList.length; i++) {
				attribute = strList[i];
				
				if (!attributeCountP1.containsKey(attribute)) {
					continue;					
				}

				posCount = 1 + attributeCountP1.get(attribute);
				negCount = 1 + attributeCountN1.get(attribute);

				posProbability += Math.log(posCount/ (double) (classCountP1 + attributeCountVals.get(attribute.split(":")[0]) + 1));
				negProbability += Math.log(negCount/ (double) (classCountN1 + attributeCountVals.get(attribute.split(":")[0]) + 1));
			}
			
			posProbability += Math.log(classCountP1/ (double) (classCountP1 + classCountN1));
			negProbability += Math.log(classCountN1/ (double) (classCountP1 + classCountN1));
			
			if (posProbability >= negProbability) {
				predictedClass = "+1";
			} else {
				predictedClass = "-1";
			}
			
			if (predictedClass.equals("+1") && predictedClass.equals(cls)) {
				truePositive += 1;
			} else if (predictedClass.equals("-1") && !predictedClass.equals(cls)){
				falseNegative += 1;
			} else if (predictedClass.equals("+1") && !predictedClass.equals(cls)){
				falsePositive += 1;
			} else {
				trueNegative += 1;
			}
		}
		
		Double accuracy = (truePositive + trueNegative) / (truePositive + falsePositive + falseNegative + trueNegative);
		Double errorRate = 	(falseNegative + falsePositive) / (truePositive + falsePositive + falseNegative + trueNegative);
		Double sensitivity = truePositive / (truePositive + falseNegative);
		Double specificity = trueNegative / (trueNegative + falsePositive);
		Double precision = truePositive / (truePositive + falsePositive);
		Double f1score = (2 * precision * sensitivity) / (precision + sensitivity);
		
		double b = 0.5;
		Double fb05 = ((1 + (b * b)) * precision * sensitivity) / ((b * b * precision) + sensitivity );
		
		b = 2;
		Double fb2 = ((1 + (b * b)) * precision * sensitivity) / ((b * b * precision) + sensitivity );
		
		System.out.println("Accuracy            : " + accuracy * 100);
		System.out.println("Error Rate          : " + errorRate * 100);
		System.out.println("Sensitivity         : " + sensitivity * 100);
		System.out.println("Specificity         : " + specificity * 100);
		System.out.println("Precision           : " + precision * 100);
		System.out.println("F-1 Score           : " + f1score);
		System.out.println("F Score(beta = 0.5) : " + fb05);
		System.out.println("F Score (beta = 2)  : " + fb2);
		
		br.close();
	}
	
	private void run() throws IOException
	{
		attributeCountP1 = new HashMap<>();
		attributeCountN1 = new HashMap<>();
		attributeCountVals = new HashMap<>();
		classCountN1 = 0;
		classCountP1 = 0;
		
		trainData();
		
		testData(trainFile);
		
		System.out.println();
		
		testData(testFile);
	}
	
	public static void main(String[] args) throws IOException
	{
		MeasuresNaiveBayes main = new MeasuresNaiveBayes(args[0], args[1]);
		main.run();
		
		System.exit(0);
	}
}
