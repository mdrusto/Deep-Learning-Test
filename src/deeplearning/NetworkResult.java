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

	public NetworkGradient getGradient(byte correctOutput, WeightMatrix[] weights) {
		//System.out.println("Correct output: " + correctOutput);
		//System.out.println("---------------------------------------------------------------");
		WeightMatrix[] finalWeightAdjustments = new WeightMatrix[layerCount - 1];
		BiasVector[] finalBiasAdjustments = new BiasVector[layerCount - 1];

		double[] previousActivationDerivatives = new double[output.length]; // del C / del a

		for (int i = 0; i < output.length; i++) {
			previousActivationDerivatives[i] = 2 * (output[i] - (correctOutput == i ? 1 : 0));
			//System.out.println(output[i]);
		}
		
		// Weight adjustments [layerCounter] are the weights between currentLayer (input) and lastLayer (output)

		for (int layerCounter = layerSnapshots.length - 2; layerCounter >= 0; layerCounter--) { // Backpropagation
			LayerSnapshot currentLayer = layerSnapshots[layerCounter];
			LayerSnapshot lastLayer = layerSnapshots[layerCounter + 1];
			
			// Compute weight derivatives
			finalWeightAdjustments[layerCounter] = new WeightMatrix(lastLayer.getSize(), currentLayer.getSize());
			for (int rowCounter = 0; rowCounter < lastLayer.getSize(); rowCounter++) {
				
				double lastActivation = lastLayer.getActivations()[rowCounter];
				double sigmoidDerivative = lastActivation * (1 - lastActivation);
				
				for (int columnCounter = 0; columnCounter < currentLayer.getSize(); columnCounter++) {
					
					// Derivative of error wrt weight[row, column]
					double currentActivation = currentLayer.getActivations()[columnCounter];
					double finalCostDerivative = currentActivation * sigmoidDerivative
							* previousActivationDerivatives[rowCounter];
					
					finalWeightAdjustments[layerCounter].initialize(rowCounter, columnCounter, finalCostDerivative); // ChainRule
				}
			}

			// Compute bias derivatives
			finalBiasAdjustments[layerCounter] = new BiasVector(lastLayer.getSize());
			for (int rowCounter = 0; rowCounter < lastLayer.getSize(); rowCounter++) {
				double lastActivation = lastLayer.getActivations()[rowCounter];
				double sigmoidDerivative = lastActivation * (1 - lastActivation);
				finalBiasAdjustments[layerCounter].initialize(rowCounter,
						sigmoidDerivative * previousActivationDerivatives[rowCounter]); // ChainRule
			}

			// Calculate this layer's activation derivatives for next layer
			if (layerCounter > 0) { // Input has no weights/biases
				double[] temp = new double[currentLayer.getSize()];

				for (int columnCounter = 0; columnCounter < currentLayer.getSize(); columnCounter++) {
					
					for (int rowCounter = 0; rowCounter < lastLayer.getSize(); rowCounter++) {
						
						double lastActivation = lastLayer.getActivations()[rowCounter];
						double sigmoidDerivative = lastActivation * (1 - lastActivation);
						
						temp[columnCounter] += weights[layerCounter].getValue(rowCounter, columnCounter)
								* sigmoidDerivative * previousActivationDerivatives[rowCounter]; // Chain rule
					}
				}
				previousActivationDerivatives = temp;
			}
		} // TODO: It ain't working properly

		return new NetworkGradient(finalWeightAdjustments, finalBiasAdjustments);
	}
	
	@Override
	public String toString() {
		StringBuilder s = new StringBuilder("{");
		for (int i = 0; i < output.length; i++) {
			if (i != 0)
				s.append(", ");
			String si = "" + Math.round(output[i] * 100) / 100.0;
			s.append(si);
		}
		return s.append('}').toString();
	}
}
