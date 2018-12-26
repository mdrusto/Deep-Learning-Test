package deeplearning;

import deeplearning.training.LayerSnapshot;

public class Layer {

	protected double[] neurons;
	private double[] zValues;
	
	public Layer(int neuronCount) {
		neurons = new double[neuronCount];
	}
	
	public void setActivations(double[] activations) {
		this.neurons = activations;
	}
	
	public int getNeuronCount() {
		return neurons.length;
	}
	
	public void setZValues(double[] zValues) {
		this.zValues = zValues;
	}
	
	public double[] getActivations() {
		return neurons;
	}
	
	public double[] getZValues() {
		return zValues;
	}
	
	public LayerSnapshot getSnapshot() {
		return new LayerSnapshot(neurons, zValues);
	}
}
