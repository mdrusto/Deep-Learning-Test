package deeplearning;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Random;
import java.util.Scanner;

public class MainClass {

	/* SETTINGS */
	private static final double INITIALIZATION_BOUND = 10.0;
	private static final double GRADIENT_MULTIPLIER = 0.1;
	private static final int TRAINING_BATCH_SIZE = 100;
	private static final double DELTA_ERROR_LIMIT = 0.1;
	private static final int SUCCESSIVE_DELTA_ERROR_CHECKS = 10;

	/* ------ */

	private static Layer[] layers;

	private static DigitImage[] images;

	public static void main(String[] useless) {

		System.out.println("Loading images...");
		images = loadImages();
		System.out.println("Successfully loaded images!");

		Scanner scanner = new Scanner(System.in);

		System.out.print("Number of layers: ");
		int nLayers = scanner.nextInt();
		System.out.println("\n");

		layers = new Layer[nLayers + 1];
		Layer lastLayer = null;

		for (int n = 0; n < nLayers; n++) {
			System.out.print("\nNumber of neurons in layer " + (n + 1) + ": ");
			layers[n] = new Layer(scanner.nextInt(), lastLayer);
			System.out.println("\n");
			lastLayer = layers[n];
		}
		layers[nLayers] = new Layer(10, lastLayer);

		long startTime = System.currentTimeMillis();
		System.out.print("Initializing weights and biases with random values...");
		initializeNetwork();
		System.out.println("\t[Done in " + (System.currentTimeMillis() - startTime) + "ms]");

		System.out.print("Training network...");
		startTime = System.currentTimeMillis();
		trainNetwork();
		System.out.println("\t[Done in " + (System.currentTimeMillis() - startTime) / 1000 + "s]");

		scanner.close();

	}

	/* Initialize weights and biases with random values */
	private static void initializeNetwork() {
		Random rand = new Random();
		int lastLayerNeuronCount = 0;
		for (int n = 0; n < layers.length; n++) {
			int neuronCount = layers[n].getNeuronCount();
			double[][] weights = new double[neuronCount][lastLayerNeuronCount];
			double[] biases = new double[neuronCount];
			for (int x = 0; x < neuronCount; x++) {
				for (int y = 0; y < lastLayerNeuronCount; y++) {
					weights[x][y] = (rand.nextDouble() - 0.5) * INITIALIZATION_BOUND * 2;
				}
				biases[x] = (rand.nextDouble() - 0.5) * INITIALIZATION_BOUND * 2;
			}
			lastLayerNeuronCount = neuronCount;

			layers[n].setWeights(weights);
			layers[n].setBiases(biases);
		}
	}

	private static void trainNetwork() {
		double networkError = 1000, previousNetworkError = 100000;
		int successiveSuccessfulErrorChecks = 0;
		for (int n = 0; successiveSuccessfulErrorChecks < SUCCESSIVE_DELTA_ERROR_CHECKS; n++) {
			//System.out.println(networkError);
			// Each iteration is one gradient step

			double[][][] finalWeightAdjustments = new double[layers.length][][];
			double[][] finalBiasAdjustments = new double[layers.length][];

			// Initialize the arrays
			for (int counter = 0; counter < layers.length; counter++) {
				double[][] weights = layers[counter].getWeights();
				double[] biases = layers[counter].getBiases();

				finalWeightAdjustments[counter] = new double[weights.length][weights[0].length];
				finalBiasAdjustments[counter] = new double[biases.length];
			}

			int[] randomTrainingDataIndices = new int[TRAINING_BATCH_SIZE];
			Random rng = new Random();
			int successfulIndexCount = 0;

			// Select training images
			do {
				int selected = rng.nextInt(60000);
				if (contains(randomTrainingDataIndices, selected)) // (not likely)
					continue; // Pick another
				else
					randomTrainingDataIndices[successfulIndexCount++] = selected;

			} while (successfulIndexCount < TRAINING_BATCH_SIZE);

			for (int x = 0; x < TRAINING_BATCH_SIZE; x++) {
				// Calculate network gradient for each image

				DigitImage image = images[randomTrainingDataIndices[x]];
				byte correctOutput = image.getLabel();
				double[] networkOutput = runNetwork(image.flatten());
				previousNetworkError = networkError;
				networkError = calculateError(networkOutput, correctOutput);

				double[] previousActivationDerivatives = new double[networkOutput.length]; // del C / del a

				for (int i = 0; i < networkOutput.length; i++) {
					previousActivationDerivatives[i] = networkOutput[i] - (correctOutput == i ? 1 : 0);
				}

				for (int layerCounter = layers.length - 1; layerCounter >= 1; layerCounter--) { // Backpropagation
					Layer currentLayer = layers[layerCounter];
					Layer lastLayer = layerCounter == layers.length - 1 ? null : layers[layerCounter + 1];

					// Compute weight derivatives
					if (layerCounter != layers.length - 1) {
						for (int i = 0; i < currentLayer.getNeuronCount(); i++) {
							for (int j = 0; j < lastLayer.getNeuronCount(); j++) {
								finalWeightAdjustments[layerCounter][j][i] = currentLayer.getActivations()[j]
										* SigmoidFunction.getDerivative(currentLayer.getZValues()[j])
										* previousActivationDerivatives[i]; // Chain rule
							}
						}
					}

					// Compute bias derivatives
					for (int i = 0; i < currentLayer.getNeuronCount(); i++) {
						finalBiasAdjustments[layerCounter][i] = SigmoidFunction
								.getDerivative(currentLayer.getZValues()[i]) * previousActivationDerivatives[i]; // ChainRule
					}

					// Calculate this layer's activation derivatives for next layer
					if (layerCounter > 1 && layerCounter < layers.length - 1) { // Input has no weights/biases
						double[] temp = new double[currentLayer.getNeuronCount()]; // For new acti derivs

						for (int m = 0; m < currentLayer.getNeuronCount(); m++) {
							for (int i = 0; i < previousActivationDerivatives.length; i++) {
								temp[m] += lastLayer.getWeights()[m][i]
										* SigmoidFunction.getDerivative(lastLayer.getZValues()[i])
										* previousActivationDerivatives[i]; // Chain rule
							}
						}
						previousActivationDerivatives = temp;
					}
				} // TODO: It ain't working properly
			}

			// Adjust network weights & biases
			for (int counter = 1; counter < layers.length - 1; counter++) {
				for (int x = 0; x < layers[counter].getNeuronCount(); x++) {
					for (int y = 0; y < layers[counter + 1].getNeuronCount(); y++) {
						layers[n].adjustWeight(x, y,
								-GRADIENT_MULTIPLIER * finalWeightAdjustments[counter][x][y] / TRAINING_BATCH_SIZE);
						// Average over batch
					}
					layers[n].adjustBias(x,
							-GRADIENT_MULTIPLIER * finalBiasAdjustments[counter][x] / TRAINING_BATCH_SIZE);
				}

			}
			if (previousNetworkError - networkError < DELTA_ERROR_LIMIT)
				successiveSuccessfulErrorChecks++;
			else
				successiveSuccessfulErrorChecks = 0;
			System.out.println(networkError);
		}
	}

	private static boolean contains(int[] array, int value) {
		for (int x : array) {
			if (x == value)
				return true;
		}
		return false;
	}

	/* networkOutput is size 10 */
	private static double calculateError(double[] networkOutput, byte correctOutput) {
		double error = 0;
		for (byte n = 0; n < networkOutput.length; n++) {
			byte correctValue = (n == correctOutput) ? (byte) 1 : 0;
			error += Math.pow(networkOutput[n] - (double) correctValue, 2);
			//System.out.println("Correct: " + correctValue + "\tNetwork: " + networkOutput[n]);
		}
		return error;
	}

	/*
	 * inputActivations is input layer size (784), return is output layer size (10)
	 */
	private static double[] runNetwork(double[] inputActivations) {
		for (int n = 0; n < layers.length; n++) {
			layers[n].update();
			if (n == layers.length - 1)
				return layers[n].getActivations();
		}
		return null;
	}

	private static DigitImage[] loadImages() {
		DigitImage[] imageArray = new DigitImage[60000];
		byte[] imageFileContents = new byte[47040016];
		byte[] labelFileContents = new byte[60008];

		try (InputStream inputStream = new FileInputStream("train-images.idx3-ubyte")) {
			for (int i = 0; i < 28; i++) {
				inputStream.read(imageFileContents);
			}
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
		try (InputStream inputStream = new FileInputStream("train-labels.idx1-ubyte")) {
			inputStream.read(labelFileContents);
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}

		for (int imageCounter = 0; imageCounter < imageArray.length; imageCounter++) {
			double[][] imageData = new double[28][28];
			byte label = labelFileContents[8 + imageCounter];
			for (int x = 0; x < 28; x++) {
				for (int y = 0; y < 28; y++) {
					imageData[x][y] = imageFileContents[16 + 784 * imageCounter + 28 * y + x];
				}
			}
			imageArray[imageCounter] = new DigitImage(imageData, label);
		}
		return imageArray;
	}
}
