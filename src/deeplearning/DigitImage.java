package deeplearning;

public class DigitImage {
	
	private double[][] contents;
	private byte label;
	
	public DigitImage(double[][] contents, byte label) {
		this.contents = contents;
		this.label = label;
	}
	
	public double[][] getContents() {
		return contents;
	}
	
	public double[] flatten() {
		double[] newArray = new double[784];
		for (int i = 0; i < 784; i++) {
			newArray[i] = contents[i / 28][i % 28];
		}
		return newArray;
	}
	
	public byte getLabel() {
		return label;
	}
}
