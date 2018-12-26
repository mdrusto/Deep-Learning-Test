package deeplearning;

public class LayerLink {
	
	private int inputLength, outputLength;
	
	private WeightMatrix weights;
	private BiasVector biases;
	
	public LayerLink(int inputLength, int outputLength) {
		this.inputLength = inputLength;
		this.outputLength = outputLength;
	}
	
	public void initialize(WeightMatrix weights, BiasVector biases) {
		this.weights = weights;
		this.biases = biases;
	}
	
	public double[] getOutput(double[] input) {
		// z = W * a + b
		if (input.length != inputLength)
			throw new IllegalArgumentException(String.format("Expected %s, got %s", inputLength, input.length));
		double[] output = new double[outputLength];
		for (int outputCounter = 0; outputCounter < outputLength; outputCounter++) {
			for (int inputCounter = 0; inputCounter < inputLength; inputCounter++) {
				output[outputCounter] += (input[inputCounter] * weights.getValue(outputCounter, inputCounter));
			}
			output[outputCounter] += biases.getValue(outputCounter);
		}
		return output;
	}
	
	public void subtractWeightsAndBiases(WeightMatrix weightAdjustments, BiasVector biasAdjustments, double multiplier) {
		weights.subtract(weightAdjustments, multiplier);
		biases.subtract(biasAdjustments, multiplier);
	}
	
	public WeightMatrix getWeights() {
		return weights;
	}
}
