package deeplearning;

import deeplearning.training.LayerSnapshot;
import deeplearning.training.NetworkGradient;

public class NetworkResult {

	private int layerCount;
	private double[] output;
	private LayerSnapshot[] layerSnapshots;

	public NetworkResult(double[] output, LayerSnapshot[] layerSnapshots) {
		this.layerCount = layerSnapshots.length;
		this.output = output;
		this.layerSnapshots = layerSnapshots;
	}

	public double[] getOutput() {
		return output;
	}

	public NetworkGradient getGradient(byte correctOutput) {
		WeightMatrix[] finalWeightAdjustments = new WeightMatrix[layerCount];
		BiasVector[] finalBiasAdjustments = new BiasVector[layerCount];

		double[] previousActivationDerivatives = new double[output.length]; // del C / del a

		for (int i = 0; i < output.length; i++) {
			previousActivationDerivatives[i] = output[i] - (correctOutput == i ? 1 : 0);
		}

		for (int layerCounter = layerSnapshots.length - 1; layerCounter >= 1; layerCounter--) { // Backpropagation
			LayerSnapshot currentLayer = layerSnapshots[layerCounter];
			LayerSnapshot lastLayer = layerCounter == layerSnapshots.length - 1 ? null
					: layerSnapshots[layerCounter + 1];

			// Compute weight derivatives
			if (layerCounter != layerSnapshots.length - 1) {
				finalWeightAdjustments[layerCounter] = new WeightMatrix(currentLayer.getSize(), lastLayer.getSize());
				for (int rowCounter = 0; rowCounter < currentLayer.getSize(); rowCounter++) {
					for (int columnCounter = 0; columnCounter < lastLayer.getSize(); columnCounter++) {
						finalWeightAdjustments[layerCounter].initialize(rowCounter, columnCounter,
								currentLayer.getActivations()[columnCounter]
										* SigmoidFunction.getDerivative(currentLayer.getZValues()[columnCounter])
										* previousActivationDerivatives[rowCounter]); // Chain rule
					}
				}
			}

			// Compute bias derivatives
			for (int rowCounter = 0; rowCounter < currentLayer.getSize(); rowCounter++) {
				finalBiasAdjustments[layerCounter].initialize(rowCounter,
						SigmoidFunction.getDerivative(currentLayer.getZValues()[rowCounter])
								* previousActivationDerivatives[rowCounter]); // ChainRule
			}

			// Calculate this layer's activation derivatives for next layer
			if (layerCounter > 1 && layerCounter < layerSnapshots.length - 1) { // Input has no weights/biases
				double[] temp = new double[currentLayer.getSize()]; // For new acti derivs

				for (int rowCounterm = 0; rowCounterm < currentLayer.getSize(); rowCounterm++) {
					for (int columnCounter = 0; columnCounter < previousActivationDerivatives.length; columnCounter++) {
						temp[rowCounterm] += lastLayer.getWeights().getValue(rowCounterm, columnCounter)
								* SigmoidFunction.getDerivative(lastLayer.getZValues()[columnCounter])
								* previousActivationDerivatives[columnCounter]; // Chain rule
					}
				}
				previousActivationDerivatives = temp;
			}
		} // TODO: It ain't working properly

		return new NetworkGradient(finalWeightAdjustments, finalBiasAdjustments);
	}
}
