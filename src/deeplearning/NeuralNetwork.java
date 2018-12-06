package deeplearning;

import java.util.Random;

import deeplearning.training.NetworkGradient;

public class NeuralNetwork {
	
	private Layer[] layers;
	/* Link 0 goes from Layer 0 to Layer 1 */
	private LayerLink[] links;
	
	public NeuralNetwork(NetworkSize size) {
		layers = new Layer[size.getLayerCount()];
		layers[0] = new InputLayer(784);
		layers[size.getLayerCount() + 1] = new OutputLayer(10);
		for (int n = 1; n < size.getLayerCount() + 1; n++) {
			layers[n] = new InternalLayer(size.getLayerLength(n));
			links[n] = new LayerLink(layers[n - 1].getNeuronCount(), layers[n].getNeuronCount());
		}
	}
	
	public void initializeNetwork(Random rand, double initializationBound) {
		int lastLayerNeuronCount = 0;
		for (int n = 0; n < layers.length; n++) {
			int neuronCount = layers[n].getNeuronCount();
			WeightMatrix weightMatrix = new WeightMatrix(neuronCount, lastLayerNeuronCount);
			BiasVector biasVector = new BiasVector(neuronCount);
			for (int rowCounter = 0; rowCounter < neuronCount; rowCounter++) {
				for (int columnCounter = 0; columnCounter < lastLayerNeuronCount; columnCounter++) {
					double initialValue = (rand.nextDouble() - 0.5) * initializationBound * 2;
					weightMatrix.initialize(rowCounter, columnCounter, initialValue);
				}
				double initialValue = (rand.nextDouble() - 0.5) * initializationBound * 2;
				biasVector.initialize(rowCounter, initialValue);
			}
			lastLayerNeuronCount = neuronCount;
			
			links[n].initialize(weightMatrix, biasVector);
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
