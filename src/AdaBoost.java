import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class AdaBoost
{
	private List<Double> tupleWeight, errorClassifierList;
	List<Map<String, Integer>> attributeCountP1, attributeCountN1;
	public List<Integer> classCountN1, classCountP1, correctCount, incorrectCount;
	Map<String, Integer> attributeCountVals;
	List<String> tuples;
	private int K_ITER = 10, SIZE;
	
	private void readTuples() throws IOException
	{
		BufferedReader br = new BufferedReader(new FileReader("files/a1a.train"));
		String str;
		
		while ((str = br.readLine()) != null) {
			tuples.add(str);
		}
		
		br.close();
	}
	
	private int returnItemBasedOnWeight()
	{
		double p = Math.random();
		double cumulativeProbability = 0.0;
		for (int i=0; i<tupleWeight.size(); i++) {
		    cumulativeProbability += tupleWeight.get(i);
		    if (p <= cumulativeProbability) {
		        return i;
		    }
		}
		
		return 0;
	}
	
	private void buildClassifier()
	{
		Map<String, Integer> attributeCountP1Local = new HashMap<>(),
							 attributeCountN1Local = new HashMap<>();

		int classCountN1Local = 0,
			classCountP1Local = 0,
			correctCountLocal = 0,
			incorrectCountLocal = 0;
		
		List<Integer> rowIds = new ArrayList<>(),
					  correctIds = new ArrayList<>(),
					  incorrectIds = new ArrayList<>();

		String str, cls, attribute, predictedClass;
		String[] strList;
		int rowId, posCount, negCount, maxVal = -1;
		double posProbability, negProbability;
		StringBuilder sb;
		
		for (int i=0; i<SIZE; i++) {
			rowId = returnItemBasedOnWeight();
			rowIds.add(rowId);
			str = tuples.get(rowId);
			strList = str.split(" ");
			cls = strList[0];
			
			if (cls.equals("-1")) {
				classCountN1Local++;
			} else {
				classCountP1Local++;
			}
			
			for (int j=1; j<strList.length; j++) {
				attribute = strList[j];
				
				if (cls.equals("+1")) {
					if (!attributeCountP1Local.containsKey(attribute)) {
						attributeCountP1Local.put(attribute, 0);
					}
					attributeCountP1Local.put(attribute, attributeCountP1Local.get(attribute) + 1);
				} else {
					if (!attributeCountN1Local.containsKey(attribute)) {
						attributeCountN1Local.put(attribute, 0);
					}
					attributeCountN1Local.put(attribute, attributeCountN1Local.get(attribute) + 1);	
				}
				
				if (Integer.parseInt(attribute.split(":")[0]) > maxVal) {
					maxVal = Integer.parseInt(attribute.split(":")[0]);
				}
			}
		}
		
		for (Entry<String, Integer> e : attributeCountVals.entrySet()) {
			int p = 0, n = 0;
			
			for (int i=1; i<=e.getValue(); i++) {
				if (!attributeCountP1Local.containsKey(e.getKey() + ":" + i)) {
					attributeCountP1Local.put(e.getKey() + ":" + i, 0);
				}
				if (!attributeCountN1Local.containsKey(e.getKey() + ":" + i)) {
					attributeCountN1Local.put(e.getKey() + ":" + i, 0);
				}
				
				p += attributeCountP1Local.get(e.getKey() + ":"  + i);
				n += attributeCountN1Local.get(e.getKey() + ":"  + i);
			}
			
			attributeCountP1Local.put(e.getKey() + ":0", classCountP1Local - p);
			attributeCountN1Local.put(e.getKey() + ":0", classCountN1Local - n);
		}
		
		for (int i=1; i<=maxVal; i++) {
			if (!attributeCountVals.containsKey(i + "")) {
				attributeCountP1Local.put(i + ":0", classCountP1Local);
				attributeCountN1Local.put(i + ":0", classCountN1Local);
				attributeCountVals.put(i + "", 0);
			}
		}
		
		for (int i=0; i<rowIds.size(); i++) {
			str = tuples.get(rowIds.get(i)).trim();
			sb = new StringBuilder(str);
			
			for (Entry<String, Integer> e : attributeCountVals.entrySet()) {
				if (!str.contains(" " + e.getKey() + ":")) {
					sb.append(" " + e.getKey() + ":0");
				}
			}
			
			strList = sb.toString().split(" ");
			cls = strList[0];
			posProbability = 1;
			negProbability = 1;
			
			for (int j=1; j<strList.length; j++) {
				attribute = strList[j];

				if (!attributeCountP1Local.containsKey(attribute)) {
					if (attribute.split(":")[1] == "0") {
						attributeCountP1Local.put(attribute, classCountP1Local);						
					} else {
						attributeCountP1Local.put(attribute, 0);
					}
				}
				
				if (!attributeCountN1Local.containsKey(attribute)) {
					if (attribute.split(":")[1] == "0") {
						attributeCountN1Local.put(attribute, classCountN1Local);						
					} else {
						attributeCountN1Local.put(attribute, 0);
					}
				}
				
				if (!attributeCountVals.containsKey(attribute.split(":")[0])) {
					attributeCountVals.put(attribute.split(":")[0], Integer.parseInt(attribute.split(":")[1]));					
				}

				posCount = 1 + attributeCountP1Local.get(attribute);
				negCount = 1 + attributeCountN1Local.get(attribute);

				posProbability += Math.log(posCount/ (double) (classCountP1Local + attributeCountVals.get(attribute.split(":")[0]) + 1));
				negProbability += Math.log(negCount/ (double) (classCountN1Local + attributeCountVals.get(attribute.split(":")[0]) + 1));
				
			}
			
			posProbability += Math.log(classCountP1Local/ (double) (classCountP1Local + classCountN1Local));
			negProbability += Math.log(classCountN1Local/ (double) (classCountP1Local + classCountN1Local));
			
			if (posProbability >= negProbability) {
				predictedClass = "+1";
			} else {
				predictedClass = "-1";
			}
			
			if (predictedClass.equals(cls)) {
				correctCountLocal += 1;
				correctIds.add(rowIds.get(i));
			} else {
				incorrectCountLocal += 1;
				incorrectIds.add(rowIds.get(i));
			}
		}
		
		if ((incorrectCountLocal / (double) (correctCountLocal + incorrectCountLocal)) > 0.5) {
			System.out.println(incorrectCountLocal / (double) (correctCountLocal + incorrectCountLocal));
			buildClassifier();
		}

		attributeCountP1.add(attributeCountP1Local);
		attributeCountN1.add(attributeCountN1Local);
		classCountP1.add(classCountP1Local);
		classCountN1.add(classCountN1Local);
		correctCount.add(correctCountLocal);
		incorrectCount.add(incorrectCountLocal);
		
		double errorClassifier = 0;
		for (int id : incorrectIds) {
			errorClassifier += tupleWeight.get(id);
		}
		errorClassifierList.add(errorClassifier);
		
		for (int id : correctIds) {
			tupleWeight.set(id, tupleWeight.get(id) * errorClassifier / (1 - errorClassifier));
		}
		
		double newWeight = 0;
		for (double w : tupleWeight) {
			newWeight += w;
		}
		
		for (int i=0; i<tupleWeight.size(); i++) {
			tupleWeight.set(i, tupleWeight.get(i) / newWeight);
		}
	}
	
	private void trainData()
	{
		for (int i=0; i<K_ITER; i++) {
			buildClassifier();
		}
	}
	
	private int testDataWithClassifier(int k, String str) throws IOException
	{
		String attribute;
		String[] strList;
		double posProbability, negProbability;
		int posCount, negCount;
		
		str = str.trim();
		StringBuilder sb = new StringBuilder(str.trim());

		for (Entry<String, Integer> e : attributeCountVals.entrySet()) {
			if (!str.contains(" " + e.getKey() + ":")) {
				sb.append(" " + e.getKey() + ":0");
			}
		}
		
		strList = sb.toString().split(" ");
		posProbability = 1;
		negProbability = 1;

		for (int i = 1; i < strList.length; i++) {
			attribute = strList[i];

			if (!attributeCountP1.get(k).containsKey(attribute)) {
				if (attribute.split(":")[1] == "0") {
					attributeCountP1.get(k).put(attribute, classCountP1.get(k));						
				} else {
					attributeCountP1.get(k).put(attribute, 0);
				}
			}
			
			if (!attributeCountN1.get(k).containsKey(attribute)) {
				if (attribute.split(":")[1] == "0") {
					attributeCountN1.get(k).put(attribute, classCountN1.get(k));						
				} else {
					attributeCountN1.get(k).put(attribute, 0);
				}
			}
			
			if (!attributeCountVals.containsKey(attribute.split(":")[0])) {
				attributeCountVals.put(attribute.split(":")[0], Integer.parseInt(attribute.split(":")[1]));					
			}

			posCount = 1 + attributeCountP1.get(k).get(attribute);
			negCount = 1 + attributeCountN1.get(k).get(attribute);

			posProbability += Math.log(posCount/ (double) (classCountP1.get(k) + attributeCountVals.get(attribute.split(":")[0]) + 1));
			negProbability += Math.log(negCount/ (double) (classCountN1.get(k) + attributeCountVals.get(attribute.split(":")[0]) + 1));
		}

		posProbability += Math.log(classCountP1.get(k)
				/ (double) (classCountP1.get(k) + classCountN1.get(k)));
		negProbability += Math.log(classCountN1.get(k)
				/ (double) (classCountP1.get(k) + classCountN1.get(k)));

		if (posProbability >= negProbability) {
			return 1;
		} else {
			return -1;
		}
	}
	
	private void testData() throws IOException
	{
		double resultPos, resultNeg;
		BufferedReader br = new BufferedReader(new FileReader("files/a1a.test"));
		String str, cls, predictedCls;
		int prediction, correctCountLocal = 0, incorrectCountLocal = 0;
		
		while ((str = br.readLine()) != null) {
			resultPos = 0;
			resultNeg = 0;
			cls = str.split(" ")[0];
			
			for (int i=0; i<K_ITER; i++) {
				prediction = testDataWithClassifier(i, str);
				if (prediction == 1) {
					resultPos += Math.log((1 - errorClassifierList.get(i)) / errorClassifierList.get(i));
				} else {
					resultNeg += Math.log((1 - errorClassifierList.get(i)) / errorClassifierList.get(i));
				}
			}	
			
			if (resultPos >= resultNeg) {
				predictedCls = "+1";
			} else {
				predictedCls = "-1";
			}
			
			if (predictedCls.equals(cls)) {
				correctCountLocal++;
			} else {
				incorrectCountLocal++;
			}
		}
		
		System.out.println(correctCountLocal);
		System.out.println(incorrectCountLocal);
		
		br.close();
	}
	
	private void run() throws IOException
	{
		tuples = new ArrayList<>();
		readTuples();
		SIZE = tuples.size();
		
		tupleWeight = new ArrayList<>();
		for (int i=0; i<tuples.size(); i++) {
			tupleWeight.add(1/(double) tuples.size());
		}
		
		attributeCountP1 = new ArrayList<>();
		attributeCountN1 = new ArrayList<>();
		attributeCountVals = new HashMap<>();
		classCountP1 = new ArrayList<>();
		classCountN1 = new ArrayList<>();
		correctCount = new ArrayList<>();
		incorrectCount = new ArrayList<>();
		errorClassifierList = new ArrayList<>();
				
		trainData();
		testData();
//		
//		double test = 0;
//		for (int i=0; i<tupleWeight.size(); i++) {
//			test += tupleWeight.get(i);
//			System.out.println(tupleWeight.get(i));
//		}
//		System.out.println(test);
	}
	
	public static void main(String[] args) throws IOException
	{
		AdaBoost main = new AdaBoost();
		main.run();
		System.exit(0);
	}
}