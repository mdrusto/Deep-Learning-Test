package deeplearning.training;

import deeplearning.WeightMatrix;

public class LayerSnapshot {

	private int size;
	private double[] activations, zValues;
	private WeightMatrix weights;

	public LayerSnapshot(double[] activations, double[] zValues, WeightMatrix weights) {
		if (activations.length != zValues.length)
			throw new IllegalArgumentException(
					String.format("Activations and Z-values array lengths must be equal, got %s and %s",
							activations.length, zValues.length));
		this.activations = activations;
		this.zValues = zValues;
		this.weights = weights;
	}

	public int getSize() {
		return size;
	}

	public double[] getActivations() {
		return activations;
	}

	public double[] getZValues() {
		return zValues;
	}
	
	public WeightMatrix getWeights() {
		return weights;
	}
}
