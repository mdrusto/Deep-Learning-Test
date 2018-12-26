package deeplearning;

import java.util.Random;

import deeplearning.training.LayerSnapshot;
import deeplearning.training.NetworkGradient;

public class NeuralNetwork {
	
	private Layer[] layers;
	/* Link 0 goes from Layer 0 to Layer 1 */
	private LayerLink[] links;
	
	public NeuralNetwork(NetworkSize size) {
		if (size == null)
			throw new IllegalArgumentException("Size cannot be null");
		layers = new Layer[size.getLayerCount()];
		links = new LayerLink[size.getLayerCount() - 1];
		layers[0] = new Layer(size.getLayerLength(0));
		layers[size.getLayerCount() - 1] = new Layer(size.getLayerLength(size.getLayerCount() - 1));
		
		for (int n = 1; n < size.getLayerCount(); n++) {
			layers[n] = new Layer(size.getLayerLength(n));
			links[n - 1] = new LayerLink(size.getLayerLength(n - 1), size.getLayerLength(n));
		}
	}
	
	public void initializeNetwork(Random rand, double initializationBound) {
		for (int n = 0; n < layers.length - 1; n++) {
			int neuronCount = layers[n + 1].getNeuronCount();
			int lastLayerNeuronCount = layers[n].getNeuronCount();
			WeightMatrix weightMatrix = new WeightMatrix(neuronCount, lastLayerNeuronCount);
			BiasVector biasVector = new BiasVector(neuronCount);
			for (int rowCounter = 0; rowCounter < neuronCount; rowCounter++) {
				for (int columnCounter = 0; columnCounter < lastLayerNeuronCount; columnCounter++) {
					double initialValue = (new Random().nextDouble() - 0.5) * initializationBound * 2;
					weightMatrix.initialize(rowCounter, columnCounter, initialValue);
				}
				double initialValue = (new Random().nextDouble() - 0.5) * initializationBound * 2;
				biasVector.initialize(rowCounter, initialValue);
			}
			links[n].initialize(weightMatrix, biasVector);
		}
	}
	
	public void adjustNetwork(NetworkGradient gradient, double multiplier) {
		for (int index = 0; index < links.length; index++) {
			links[index].subtractWeightsAndBiases(gradient.getWeightAdjustments(index), gradient.getBiasVectorAdjustments(index), multiplier);
		}
	}
	
	public NetworkResult runNetwork(double[] input) {
		LayerSnapshot[] snapshots = new LayerSnapshot[layers.length];
		Layer firstLayer = layers[0];
		firstLayer.setActivations(input);
		snapshots[0] = firstLayer.getSnapshot();
		for (int index = 1; index < layers.length; index++) {
			double[] zValues = links[index - 1].getOutput(input);
			layers[index].setZValues(zValues);
			double[] output = SigmoidFunction.apply(zValues);
			if (index == 2)
				for (int x = 0; x < output.length; x++) {
					//System.out.println("zValue" + (x + 1) + ": " + zValues[x]);
					System.out.println("Activation " + (x + 1) + ": " + output[x]);
				}
			layers[index].setActivations(output);
			input = output;
			snapshots[index] = layers[index].getSnapshot();
		}
		return new NetworkResult(input, snapshots);
	}
	
	public WeightMatrix[] getWeights() {
		WeightMatrix[] weights = new WeightMatrix[layers.length - 1];
		for (int x = 0; x < layers.length - 1; x++) {
			weights[x] = links[x].getWeights();
		}
		return weights;
	}
}
