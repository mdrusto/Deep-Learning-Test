package deeplearning;

public class Layer {

	protected double[] neurons;
	private double[] zValues;
	
	public Layer(int neuronCount) {
		neurons = new double[neuronCount];
	}
	
	public int getNeuronCount() {
		return neurons.length;
	}
	
	public void setZValues() {
		
	}
	
	public double[] getActivations() {
		return neurons;
	}
	
	public double[] getZValues() {
		return zValues;
	}
	
	public double[] getOutput() {
		
	}
}
