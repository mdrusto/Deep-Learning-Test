package deeplearning;

public class BiasVector {
	
	private double[] biases;
	
	public BiasVector(int length) {
		biases = new double[length];
	}
	
	public void initialize(int row, double value) {
		biases[row] = value;
	}
	
	public double getValue(int index) {
		return biases[index];
	}
	
	public void subtract(BiasVector adjustments, double multiplier) {
		for (int y = 0; y < biases.length; y++) {
			biases[y] -= adjustments.getValue(y) * multiplier;
		}
	}
	
	public void divideBy(double divisor) {
		for (int y = 0; y < biases.length; y++) {
			biases[y] = biases[y] / divisor;
		}
	}
}
