package deeplearning;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Random;
import java.util.Scanner;

import deeplearning.training.NetworkGradient;

public class MainClass {

	/* SETTINGS */
	private static final double INITIALIZATION_BOUND = 1.0;
	private static final double GRADIENT_MULTIPLIER = 0.01;
	private static final int TRAINING_BATCH_SIZE = 100;
	private static final double DELTA_ERROR_LIMIT = 0.1;
	private static final int SUCCESSIVE_DELTA_ERROR_CHECKS = 10;

	/* ------ */

	private static DigitImage[] images;

	public static void main(String[] useless) {

		System.out.println("Loading images...");
		images = loadImages();
		System.out.println("Successfully loaded images!");

		NetworkSize size = askForNetworkSize();
		NeuralNetwork network = new NeuralNetwork(size);
		
		long startTime = System.currentTimeMillis();
		System.out.print("Initializing weights and biases with random values...");
		network.initializeNetwork(new Random(), INITIALIZATION_BOUND);
		System.out.println("\t[Done in " + (System.currentTimeMillis() - startTime) + "ms]");

		System.out.print("Training network...");
		startTime = System.currentTimeMillis();
		trainNetwork(network);
		System.out.println("\t[Done in " + (System.currentTimeMillis() - startTime) / 1000 + "s]");
	}

	private static void trainNetwork(NeuralNetwork network) {
		double networkError = 1000, previousNetworkError = 100000;
		int successiveSuccessfulErrorChecks = 0;
		while (successiveSuccessfulErrorChecks < SUCCESSIVE_DELTA_ERROR_CHECKS) {
			//System.out.println(networkError);
			// Each iteration is one gradient step

			int[] randomTrainingDataIndices = selectImageIndices();
			
			NetworkGradient averageGradient = null;
			
			for (int x = 0; x < TRAINING_BATCH_SIZE; x++) {
				// Calculate network gradient for each image

				DigitImage image = images[randomTrainingDataIndices[x]];
				byte correctOutput = image.getLabel();
				
				double[] networkInput = image.flatten();
				
				NetworkResult result = network.runNetwork(networkInput);
				NetworkGradient gradient = result.getGradient(correctOutput, network.getWeights());
				System.out.println(result + ", Correct = " + (correctOutput + 1));
				
				gradient.divideBy(TRAINING_BATCH_SIZE);
				
				if (averageGradient == null)
					averageGradient = gradient;
				else
					averageGradient.subtract(gradient);
				
				previousNetworkError = networkError;
				networkError = calculateError(result.getOutput(), correctOutput);
				
			}
			//System.exit(0);

			// Adjust network weights & biases
			network.adjustNetwork(averageGradient, GRADIENT_MULTIPLIER);
			
			System.out.println(networkError);
			
			if (Math.abs(previousNetworkError - networkError) < DELTA_ERROR_LIMIT)
				successiveSuccessfulErrorChecks++;
			else
				successiveSuccessfulErrorChecks = 0;
		}
	}
	
	private static int[] selectImageIndices() {
		int[] indices = new int[TRAINING_BATCH_SIZE];
		Random rng = new Random();
		int successfulIndexCount = 0;

		// Select training images
		do {
			int selected = rng.nextInt(60000);
			if (contains(indices, selected)) // (not likely)
				continue; // Pick another
			else
				indices[successfulIndexCount++] = selected;

		} while (successfulIndexCount < TRAINING_BATCH_SIZE);
		
		return indices;
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
			byte label = labelFileContents[8 + imageCounter]; //heres the error
			for (int x = 0; x < 28; x++) {
				for (int y = 0; y < 28; y++) {
					imageData[x][y] = (imageFileContents[16 + 784 * imageCounter + 28 * y + x] & 0xFF) / 256.0;
				}
			}
			imageArray[imageCounter] = new DigitImage(imageData, label);
		}
		return imageArray;
	}
	
	private static NetworkSize askForNetworkSize() {
		Scanner scanner = new Scanner(System.in);

		System.out.print("Number of internal layers: ");
		int internalLayerCount = scanner.nextInt();
		System.out.println("\n");
		
		int[] internalLayerSizes = new int[internalLayerCount];
		
		System.out.print("Size of input layer: ");
		int inputLayerSize = scanner.nextInt();
		System.out.println("\n");
		
		for (int n = 0; n < internalLayerCount; n++) {
			System.out.print("\nSize of internal layer " + (n + 1) + ": ");
			internalLayerSizes[n] = scanner.nextInt();
			System.out.println("\n");
		}
		
		System.out.print("Size of output layer: ");
		int outputLayerSize = scanner.nextInt();
		System.out.println("\n");
		
		scanner.close();
		return new NetworkSize(inputLayerSize, internalLayerSizes, outputLayerSize);
	}
}
