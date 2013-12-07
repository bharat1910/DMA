import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

public class NaiveBayes
{
	Map<String, Integer> attributeCountP1, attributeCountN1;
	Map<String, Integer> attributeCountVals;
	public int classCountN1, classCountP1, correctCount, incorrectCount;
	String trainFile, testFile;
	
	public NaiveBayes(String f1, String f2)
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
	
	private void testData() throws IOException
	{
		BufferedReader br = new BufferedReader(new FileReader(testFile));
		String str, cls, attribute, predictedClass;
		String[] strList;
		double posProbability, negProbability;
		int posCount, negCount;
		StringBuilder sb;
		
		while ((str = br.readLine()) != null) {
			posProbability = 1;
			negProbability = 1;
			str = str.trim();
			sb = new StringBuilder(str.trim());

			for (Entry<String, Integer> e : attributeCountVals.entrySet()) {
				if (!str.contains(" " + e.getKey() + ":")) {
					sb.append(" " + e.getKey() + ":0");
				}
			}
			
			strList = sb.toString().split(" ");
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
			
			if (predictedClass.equals(cls)) {
				correctCount += 1;
			} else {
				incorrectCount += 1;
			}
		}
		
		System.out.println(correctCount);
		System.out.println(incorrectCount);
		
		br.close();
	}
	
	private void run() throws IOException
	{
		attributeCountP1 = new HashMap<>();
		attributeCountN1 = new HashMap<>();
		attributeCountVals = new HashMap<>();
		classCountN1 = 0;
		classCountP1 = 0;
		correctCount = 0;
		incorrectCount = 0;
		
		trainData();
		
		testData();
	}
	
	public static void main(String[] args) throws IOException
	{
		NaiveBayes main = new NaiveBayes(args[0], args[0]);
		main.run();
		
		main = new NaiveBayes(args[0], args[1]);
		main.run();
		
		System.exit(0);
	}
}
