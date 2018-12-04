package deeplearning;

public class SigmoidFunction {
	
	public static double apply(double input) {
		return (1 / (1 + Math.exp(-input)));
	}
	
	public static double getDerivative(double input) {
		return (Math.exp(-input) / Math.pow(1 + Math.exp(-input), 2));
	}
}
