package deeplearning;

import deeplearning.training.NetworkGradient;

public class NeuralNetwork {
	
	private Layer[] layers;
	/* Link 0 goes from Layer 0 to Layer 1 */
	private LayerLink[] links;
	
	public NeuralNetwork(NetworkSize size) {
		layers = new Layer[size.getLayerCount() + 2];
		layers[0] = new InputLayer(784);
		layers[size.getLayerCount() + 1] = new OutputLayer(10);
		for (int n = 1; n < size.getLayerCount() + 1; n++) {
			layers[n] = new InternalLayer(size.getLayerLength(n));
			links[n] = new LayerLink(layers[n - 1].getNeuronCount(), layers[n].getNeuronCount());
		}
	}
	
	public void adjustNetwork(NetworkGradient gradient, double multiplier) {
		for (int index = 0; index < links.length; index++) {
			links[index].adjustWeightsAndBiases(gradient.getWeightAdjustments(index), gradient.getBiasVectorAdjustments(index), multiplier);
		}
	}
	
	private double[] runStep(int index, double[] input) {
		return links[index].getOutput(input);
	}
	
	public NetworkResult runNetwork(double[] input) {
		for (int index = 0; index < layers.length; index++) {
			
		}
	}
}
