package deeplearning;

public class BiasVector {
	
	private boolean isInitialized = false;
	
	private double[] biases;
	
	public BiasVector(int length) {
		biases = new double[length];
	}
	
	public void initialize(int row, double value) {
		if (isInitialized)
			throw new UnsupportedOperationException("Vector has already been initialized");
		isInitialized = true;
		biases[row] = value;
	}
	
	public double getValue(int index) {
		return biases[index];
	}
	
	public void add(BiasVector adjustments, double multiplier) {
		for (int y = 0; y < biases.length; y++) {
			biases[y] += adjustments.getValue(y) * multiplier;
		}
	}
}
