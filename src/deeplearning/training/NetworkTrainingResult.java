package deeplearning.training;

import deeplearning.NetworkResult;

public class NetworkTrainingResult extends NetworkResult {
	
	private LayerSnapshot[] layerSnapshots;
	
	public NetworkTrainingResult(double[] output, LayerSnapshot[] layerSnapshots) {
		super(output);
		this.layerSnapshots = layerSnapshots;
	}
	
	public NetworkGradient getGradient() {
		
	}
}
