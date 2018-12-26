package deeplearning.training;

public class LayerSnapshot {

	private int size;
	private double[] activations, zValues;

	public LayerSnapshot(double[] activations, double[] zValues) {
		if (zValues != null && activations.length != zValues.length)
			throw new IllegalArgumentException(
					String.format("Activations and Z-values array lengths must be equal, got %s and %s",
							activations.length, zValues.length));
		this.size = activations.length;
		this.activations = activations;
		this.zValues = zValues;
	}

	public int getSize() {
		return size;
	}

	public double[] getActivations() {
		return activations;
	}

	public double[] getZValues() {
		return zValues;
	}
}
