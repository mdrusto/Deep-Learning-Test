package deeplearning.training;

import deeplearning.BiasVector;
import deeplearning.WeightMatrix;

public class NetworkGradient {
	
	private WeightMatrix[] weightMatrixAdjustments;
	private BiasVector[] biasVectorAdjustments;
	
	public NetworkGradient(WeightMatrix[] weightMatrixAdjustments, BiasVector[] biasVectorAdjustments) {
		if (weightMatrixAdjustments.length != biasVectorAdjustments.length)
			throw new IllegalArgumentException("Weight and bias arrays must be equal size");
		this.weightMatrixAdjustments = weightMatrixAdjustments;
		this.biasVectorAdjustments = biasVectorAdjustments;
	}
	
	public WeightMatrix getWeightAdjustments(int index) {
		return weightMatrixAdjustments[index];
	}
	
	public BiasVector getBiasVectorAdjustments(int index) {
		return biasVectorAdjustments[index];
	}
	
	public void subtract(NetworkGradient addend) {
		for (int i = 0; i < weightMatrixAdjustments.length; i++) {
			weightMatrixAdjustments[i].subtract(addend.getWeightAdjustments(i), 1);
			biasVectorAdjustments[i].subtract(addend.getBiasVectorAdjustments(i), 1);
		}
	}
	
	public void divideBy(double divisor) {
		for (WeightMatrix weightMatrix: weightMatrixAdjustments) {
			weightMatrix.divideBy(divisor);
		}
		for (BiasVector biasVector: biasVectorAdjustments) {
			biasVector.divideBy(divisor);
		}
	}
}
