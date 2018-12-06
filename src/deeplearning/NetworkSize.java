package deeplearning;

public class NetworkSize {
	
	private int networkLength;
	
	private int inputLayerLength;
	private int[] internalLayerLengths;
	private int outputLayerLength;
	
	public NetworkSize(int inputLayerLength, int[] internalLayerLengths, int outputLayerLength) {
		this.inputLayerLength = inputLayerLength;
		this.internalLayerLengths = internalLayerLengths;
		this.outputLayerLength = outputLayerLength;
		
		networkLength = internalLayerLengths.length + 2;
	}
	
	public int getLayerLength(int layer) {
		if (layer < 0 || layer >= networkLength)
			throw new IllegalArgumentException("Layer index out of bounds: " + layer);
		if (layer == 0)
			return inputLayerLength;
		else if (layer == networkLength - 1)
			return outputLayerLength;
		else
			return internalLayerLengths[layer - 1];
	}
	
	public int getLayerCount() {
		return networkLength;
	}
}
