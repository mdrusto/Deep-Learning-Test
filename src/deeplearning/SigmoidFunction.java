package deeplearning;

public class SigmoidFunction {
	
	private static double apply(double input) {
		return (1 / (1 + Math.exp(-input)));
	}
	
	public static double[] apply(double[] input) {
		double[] output = new double[input.length];
		for (int i = 0; i < input.length; i++) {
			output[i] = apply(input[i]);
		}
		return output;
	}
}
