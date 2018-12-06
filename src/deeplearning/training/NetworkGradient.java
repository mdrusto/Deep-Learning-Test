package deeplearning.training;

import deeplearning.BiasVector;
import deeplearning.WeightMatrix;

public class NetworkGradient {
	
	private WeightMatrix[] weightMatrixAdjustments;
	private BiasVector[] biasVectorAdjustments;
	
	public NetworkGradient(WeightMatrix[] weightMatrixAdjustments, BiasVector[] biasVectorAdjustments) {
		this.weightMatrixAdjustments = weightMatrixAdjustments;
		this.biasVectorAdjustments = biasVectorAdjustments;
	}
	
	public WeightMatrix getWeightAdjustments(int index) {
		return weightMatrixAdjustments[index];
	}
	
	public BiasVector getBiasVectorAdjustments(int index) {
		return biasVectorAdjustments[index];
	}
	
	public void add(NetworkGradient addend) {
		
	}
	
	public void divideBy(double divisor) {
		
	}
}
