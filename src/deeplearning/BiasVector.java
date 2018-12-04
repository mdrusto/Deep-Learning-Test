package deeplearning;

public class BiasVector {
	
	private boolean isInitialized = false;
	
	private double[] biases;
	
	public BiasVector(int length) {
		biases = new double[length];
	}
	
	public void initialize(double[] values) {
		if (isInitialized)
			throw new UnsupportedOperationException("Vector has already been initialized");
		isInitialized = true;
		if (values.length != biases.length)
			throw new IllegalArgumentException(String.format("Expected %s, got %s", biases.length, values.length));
		for (int x = 0; x < biases.length; x++) {
			biases[x] = values[x];
		}
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
