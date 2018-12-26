package deeplearning;

public class WeightMatrix {
	
	private WeightMatrixRow[] rows;
	
	public WeightMatrix(int length, int width) {
		if (length <= 0 || width <= 0)
			throw new IllegalArgumentException(String.format("Expected positive, got length=%s, width=%s", length, width));
		rows = new WeightMatrixRow[length];
		for (int x = 0; x < length; x++) {
			rows[x] = new WeightMatrixRow(width);
		}
	}
	
	public void initialize(int row, int column, double value) {
		checkLength(row);
		checkWidth(column);
		rows[row].initialize(column, value);
	}
	
	public double getValue(int row, int column) {
		checkLength(row);
		checkWidth(column);
		return rows[row].getValue(column);
	}
	
	public void subtract(WeightMatrix adjustments, double multiplier) {
		if (rows.length != adjustments.rows.length)
			throw new IllegalArgumentException("expected matrix height: " + rows.length + ", got: " + adjustments.rows.length);
		if (rows[0].getLength() != adjustments.rows[0].getLength())
			throw new IllegalArgumentException("expected matrix width: " + rows[0].getLength() + ", got: " + adjustments.rows[0].getLength());
		for (int y = 0; y < rows.length; y++) {
			WeightMatrixRow row = rows[y];
			for (int x = 0; x < rows[0].getLength(); x++) {
				row.subtract(x, adjustments.getValue(y, x) * multiplier);
			}
		}
	}
	
	public void divideBy(double divisor) {
		for (WeightMatrixRow row: rows) {
			row.divideBy(divisor);
		}
	}
	
	private void checkWidth(int width) {
		if (width >= rows[0].getLength())
			throw new IllegalArgumentException("column # must be less than " + rows.length);
	}
	
	private void checkLength(int length) {
		if (length >= rows.length)
			throw new IllegalArgumentException("row # must be less than " + rows.length + ", got " + length);
	}
	
	private class WeightMatrixRow {
		
		private double[] weights;
		
		private WeightMatrixRow(int length) {
			weights = new double[length];
		}
		
		private void initialize(int column, double value) {
			weights[column] = value;
		}
		
		private double getValue(int index) {
			if (index >= weights.length)
				throw new IllegalArgumentException("index must be less than " + weights.length);
			return weights[index];
		}
		
		private void subtract(int index, double value) {
			weights[index] -= value;
		}
		
		private void divideBy(double divisor) {
			for (int x = 0; x < weights.length; x++) {
				weights[x] = weights[x] / divisor;
			}
		}
		
		private int getLength() {
			return weights.length;
		}
	}
}
